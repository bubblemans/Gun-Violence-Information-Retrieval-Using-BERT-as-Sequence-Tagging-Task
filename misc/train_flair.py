import argparse
import os
import math

import pandas as pd
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence


class GunViolenceDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


class Linear(nn.Module):
    def __init__(self, num_labels):
        super(Linear, self).__init__()
        self.embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward'),
                                        FlairEmbeddings('news-backward'),
                                       ])
        self.classifier = nn.Linear(4196, num_labels)

    def forward(self, input_batch, labels=None):
        embeddings = torch.tensor([]).to('cuda')
        sentence = None # TODO: remove this
        for input_sentence in input_batch:
            sentence = Sentence(input_sentence)
            embedding = self.embeddings.embed(sentence)
            sentence_embedding = torch.tensor([]).to('cuda')

            # concatanate all word representations together as a 2d tensor
            for token in sentence:
                # token.embedding is a 1d tensor that represents a single word
                sentence_embedding = torch.cat((sentence_embedding, torch.unsqueeze(token.embedding, 0)))

            # concatanate all sentence representations in a batch as 3d tensor
            embeddings = torch.cat((embeddings, torch.unsqueeze(sentence_embedding, 0)))

        logits = self.classifier(embeddings)
        return logits, embeddings, sentence  # TODO: only return logits


def _handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', type=str, required=True, help='Input data directory that contains train.csv and eval.csv')
    parser.add_argument('--lr', default=1e-4, type=float, required=True, help='learning rate')
    parser.add_argument('--cuda_available', default=True, type=bool, required=False, help='decide whether to use GPU')
    parser.add_argument('--epochs', default=1, type=int, required=False, help='the number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='the number of batches')
    parser.add_argument('--max_seq_length', default=256, type=int, required=False, help='the number of max sequence length')
    parser.add_argument('--model', default='Linear', type=str, required=False, help='Linear, LSTM, BiLSTM')
    parser.add_argument('--is_balance', default=True, type=bool, required=False, help='choose to use balance data or unbalanced data')
    return parser.parse_args()


def convert_labels(tokens, batch, max_seq_length):
    batch_labels = []
    batch_tokens = []

    for token, sentence in zip(tokens, batch):
        sentence_labels = []
        sentence_tokens = token.split()

        for word in sentence.split():
            if word == 'B-SHOOTER':
                sentence_labels.append(0)
            elif word == 'I-SHOOTER':
                sentence_labels.append(1)
            else:
                sentence_labels.append(2)

        if len(sentence_labels) > max_seq_length:
            sentence_labels = sentence_labels[:max_seq_length]
            sentence_tokens = sentence_tokens[:max_seq_length]
        elif len(sentence_labels) < max_seq_length:
            sentence_labels += [2] * (max_seq_length - len(sentence_labels))
            sentence_tokens += ['#'] * (max_seq_length - len(sentence_tokens))

        batch_labels.append(sentence_labels)
        batch_tokens.append(' '.join(sentence_tokens))

    return batch_tokens, torch.tensor(batch_labels).to('cuda')


def train(train_X, train_Y, learning_rate, cuda_available, epochs, model_type, is_balance, batch_size, max_seq_length):

    training_set = GunViolenceDataset(train_X, train_Y)
    training_generator = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
    )
    iter_in_one_epoch = len(train_X) // batch_size

    model = Linear(3)

    if cuda_available:
        model.to('cuda')  # move data onto GPU

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(1, epochs + 1):
        with tqdm.tqdm(training_generator, unit="batch") as tepoch:
            for i, (train_x, train_y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                tokens, labels = convert_labels(train_x, train_y, max_seq_length)

                # forward pass
                y_pred, embeddings, sentence = model(tokens, labels)
                y_pred = y_pred.permute(0, 2, 1)

                # calculate loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(y_pred, labels)
                losses.append((epoch + i / iter_in_one_epoch, loss.item()))

                # display loss
                tepoch.set_postfix(loss="{:.4f}".format(loss.item()))

                # zero out gradients
                optimizer.zero_grad()

                # backward pass
                loss.backward()

                # update parameters
                optimizer.step()

    torch.save(model, 'output/model')

    # with open('./data/{}_train_loss.txt'.format(model_type), 'w') as wf:
    #     for i, loss in losses:
    #         wf.write(str(i) + ' ' + str(loss) + '\n')

    return model


def evaluate(model, evaluate_X, evaluate_Y, cuda_available, batch_size, max_seq_length):

    def _get_prediction(normalized_probs):
        # classify B, I, O based on probabilities
        labels = []
        for sample_prob in normalized_probs:
            max_prob = -math.inf
            label = None
            for i, prob in enumerate(sample_prob):
                if max_prob < prob:
                    max_prob = prob
                    label = i
            labels.append(label)
        return labels

    model.eval()
    num_samples = len(evaluate_X)
    evaluate_set = GunViolenceDataset(evaluate_X, evaluate_Y)
    evaluate_generator = DataLoader(
        evaluate_set,
        batch_size=1,
        shuffle=True,
    )
    num_of_tp = num_of_fn = num_of_fp = num_of_tn = 0

    for i, (evaluate_x, evaluate_y) in enumerate(evaluate_generator):
        tokens, labels = convert_labels(evaluate_x, evaluate_y, max_seq_length)

        with torch.no_grad():
            y_pred, embeddings, sentence = model(tokens, labels)
            y_pred_for_loss = y_pred.permute(0, 2, 1)
            y_pred = y_pred[0]  # because batch size is 1, we just take the 1st row
            normalized_probs = nn.functional.softmax(y_pred, dim=1)
            results = _get_prediction(normalized_probs)

            # get the real target
            original = ''
            for i, (x, y) in enumerate(zip(evaluate_x[0].split(), evaluate_y[0].split())):
                if y[0] == 'B':
                    original = x + ' '
                    index = i
                    while index + 1 < len(evaluate_y[0].split()) and evaluate_y[0].split()[index + 1][0] == 'I':
                        original += '{} '.format(evaluate_x[0].split()[index + 1])
                        index += 1
                    break
            original = original.strip()

            probabilities = []
            predictions = []
            prediction = []
            for token, tag, prob in zip(tokens[0].split(), results, normalized_probs):
                if tag == 0:
                    # tag == 'B'
                    probabilities.append(prob)

                    if len(prediction) != 0:
                        predictions.append(prediction)
                        prediction = []
                    prediction.append(token)
                elif tag == 1:
                    # tag == 'I'
                    prediction.append(token)
            if len(prediction) != 0:
                predictions.append(prediction)

            # one sentence might generate multiple targets, eg. shooters or victims
            # we need to pick the most possible one, which is the one has the highest probability in 'B' tag
            max_prob = -math.inf
            max_prob_ind = 0
            for i, prob in enumerate(probabilities):
                if max_prob < prob[0]:
                    max_prob_ind = i
                    max_prob = prob[0]

            # calculate true positive, false positive, true negative, false negative
            result = ''
            if len(predictions) != 0:
                result = ' '.join(predictions[max_prob_ind])
                if result == original:
                    num_of_tp += 1
                else:
                    num_of_fp += 1
            else:
                if original.strip() != '':
                    num_of_fn += 1
                else:
                    num_of_tn += 1

            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(y_pred_for_loss, labels)
            # losses.append((epoch + i / iter_in_one_epoch, loss.item()))

    accuracy = num_of_tp/num_samples if num_samples != 0 else 0
    precision = num_of_tp/(num_of_tp + num_of_fp) if num_of_tp + num_of_fp != 0 else 0
    recall = num_of_tp/(num_of_tp + num_of_fn) if num_of_tp + num_of_fn != 0 else 0

    print('tp:', num_of_tp)
    print('tn:', num_of_tn)
    print('fp:', num_of_fp)
    print('fn:', num_of_fn)

    print('total:', num_samples)
    print('correct:', num_of_tp)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    f1 = 2 * precision * recall / (precision + recall)
    print('F1: {}'.format(f1))


def get_data(filename):
    df = pd.read_csv(filename)
    texts = df['texts'].tolist()
    labels = df['labels'].tolist()
    return texts, labels


if __name__ == '__main__':
    args = _handle_arguments()

    train_X, train_Y = get_data('new_data/new_train.csv')
    dev_X, dev_Y = get_data('new_data/new_dev.csv')
    train_X += dev_X
    train_Y += dev_Y
    model = train(train_X, train_Y, args.lr, args.cuda_available, args.epochs, args.model, args.is_balance, args.batch_size, args.max_seq_length)
    # model = torch.load('output/model')

    test_X, test_Y = get_data('new_data/new_test.csv')
    eval_results = evaluate(model, test_X, test_Y, args.cuda_available, args.batch_size, args.max_seq_length)
