import argparse
import os
import math

import pandas as pd
import tqdm

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Digits


label_mapping = {
    'B': 0,
    'I': 1,
    'O': 2
}

ACCURACY = []
PRECISION = []
RECALL = []


class GunViolenceDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


class BERT_Linear(nn.Module):
    def __init__(self, num_labels):
        super(BERT_Linear, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        config.max_position_embeddings = 1024
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, tokens_tensor, segments_tensors, labels=None):
        bert_output = self.bert(tokens_tensor, token_type_ids=segments_tensors)
        last_hidden_state = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output

        logits = self.classifier(last_hidden_state)
        return logits


class BERT_LSTM(nn.Module):
    def __init__(self, num_labels):
        super(BERT_LSTM, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        config.max_position_embeddings = 1024
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.lstm = nn.LSTM(768, 768)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, tokens_tensor, segments_tensors, labels=None):
        bert_output = self.bert(tokens_tensor, token_type_ids=segments_tensors)
        last_hidden_state = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output

        lstm_out, _ = self.lstm(last_hidden_state)

        logits = self.classifier(lstm_out)
        return logits


class BERT_BiLSTM(nn.Module):
    def __init__(self, num_labels):
        super(BERT_BiLSTM, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        config.max_position_embeddings = 1024
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.lstm = nn.LSTM(768, 768, bidirectional=True)
        self.classifier = nn.Linear(768, num_labels)
        # self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, tokens_tensor, segments_tensors, labels=None):
        bert_output = self.bert(tokens_tensor, token_type_ids=segments_tensors)
        last_hidden_state = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output

        lstm_out, _ = self.lstm(last_hidden_state)
        lstm_out = lstm_out[:, :, :768] + lstm_out[:, :, 768:]

        logits = self.classifier(lstm_out)
        return logits


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


def train(train_X, train_Y, learning_rate, cuda_available, epochs, model_type, is_balance, batch_size, max_seq_length):

    training_set = GunViolenceDataset(train_X, train_Y)
    training_generator = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
    )
    iter_in_one_epoch = len(train_X) // batch_size

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased') # cased!
    model = None
    if model_type == 'LSTM':
        model = BERT_LSTM(3)
    elif model_type == 'BiLSTM':
        model = BERT_BiLSTM(3)
    else:
        model = BERT_Linear(3)  # 3 different labels: B, I, O

    if cuda_available:
        model.to('cuda')  # move data onto GPU

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(1, epochs + 1):
        with tqdm.tqdm(training_generator, unit="batch") as tepoch:
            for i, (train_x, train_y) in enumerate(tepoch):
                tepoch.set_description("Epoch {}".format(epoch))

                # prepare model input
                tokens, labels = convert_examples_to_features(train_x, train_y, tokenizer, max_seq_length)
                indexed_tokens = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
                segments_ids = [[0] * len(indexed_token) for indexed_token in indexed_tokens]

                if cuda_available:
                    segments_tensors = torch.tensor(segments_ids).to('cuda')
                    tokens_tensor = torch.tensor(indexed_tokens).to('cuda')
                    labels = torch.tensor(labels).to('cuda')
                else:
                    segments_tensors = torch.tensor(segments_ids)
                    tokens_tensor = torch.tensor(indexed_tokens)
                    labels = torch.tensor(labels)

                # forward pass
                y_pred = model(tokens_tensor, segments_tensors, labels)
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

    return model, tokenizer


def convert_examples_to_features(x_batch, y_batch, tokenizer, max_seq_length):
    """
    Convert preprocessed data with tags to BERT's input format

    ex: After preprocessed
    Text: [", I, am, Alvin, ., "] (original: "I am Alvin.")
    Tags:  O  O   O    B    O  O

    After converted
    tokens: [", I, am, Al, ##vin, ., "]
    labels:  O  O   O   B    I    O  O
    """
    token_batch = []
    label_batch = []
    for train_x, train_y in zip(x_batch, y_batch):
        text, labels_org = train_x, train_y.strip().split()[:max_seq_length]
        tokens = tokenizer.tokenize(text, truncation=True, max_length=max_seq_length)

        labels = []
        token_bias_num = 0
        word_num = 0
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                if labels_org[i - 1 - token_bias_num][0] in ['O', 'I']:
                    label = label_mapping[labels_org[i - 1 - token_bias_num][0]]
                    labels.append(label)
                else:
                    labels.append(1)  # 1 is I
                token_bias_num += 1
            else:
                word_num += 1
                label = label_mapping[labels_org[i - token_bias_num][0]]
                labels.append(label)

        # manually pad tokens and labels if their lengths are over the max sequence length
        if len(tokens) < max_seq_length:
            tokens += [''] * (max_seq_length - len(tokens))
            labels += [label_mapping['O']] * (max_seq_length - len(labels))

        token_batch.append(tokens)
        label_batch.append(labels)

    return token_batch, label_batch


def evaluate(model, evaluate_X, evaluate_Y, tokenizer, cuda_available, batch_size, max_seq_length, model_type, lr, epochs):

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
    # losses = []

    for i, (evaluate_x, evaluate_y) in enumerate(evaluate_generator):
        tokens, labels = convert_examples_to_features(evaluate_x, evaluate_y, tokenizer, max_seq_length)

        indexed_tokens = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        segments_ids = [[0] * len(indexed_token) for indexed_token in indexed_tokens]

        if cuda_available:
            segments_tensors = torch.tensor(segments_ids).to('cuda')
            tokens_tensor = torch.tensor(indexed_tokens).to('cuda')
            labels = torch.tensor(labels).to('cuda')
        else:
            segments_tensors = torch.tensor(segments_ids)
            tokens_tensor = torch.tensor(indexed_tokens)
            labels = torch.tensor(labels)

        with torch.no_grad():
            y_pred = model(tokens_tensor, segments_tensors, labels)
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
            # original = ''
            # for x, y in zip(evaluate_x[0].split(), evaluate_y[0].split()):
            #     if y[0] in ['B', 'I']:
            #         original += '{} '.format(x)
            # original = original.strip()

            probabilities = []
            predictions = []
            prediction = []
            for token, tag, prob in zip(tokens[0], results, normalized_probs):
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
                result = tokenizer.convert_tokens_to_string(predictions[max_prob_ind])
                # print('original:', original)
                # print('result:', result)
                # print()
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

    accuracy = (num_of_tp + num_of_tn) /num_samples if num_samples != 0 else 0
    precision = num_of_tp/(num_of_tp + num_of_fp) if num_of_tp + num_of_fp != 0 else 0
    recall = num_of_tp/(num_of_tp + num_of_fn) if num_of_tp + num_of_fn != 0 else 0

    with open('./data/victim/nodev_{}_{}_{}_{}_{}.txt'.format(model_type, lr, epochs, batch_size, max_seq_length), 'w') as wf:
        wf.write('tp: {}\n'.format(num_of_tp))
        wf.write('tn: {}\n'.format(num_of_tn))
        wf.write('fp: {}\n'.format(num_of_fp))
        wf.write('fn: {}\n'.format(num_of_fn))

        wf.write('total: {}\n'.format(num_samples))
        wf.write('correct: {}\n'.format(num_of_tp))
        wf.write('accuracy: {}\n'.format(accuracy))
        wf.write('precision: {}\n'.format(precision))
        wf.write('recall: {}\n'.format(recall))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        wf.write('F1: {}\n'.format(f1))

    if accuracy != 0:
        ACCURACY.append(accuracy)
    if precision != 0:
        PRECISION.append(precision)
    if recall != 0:
        RECALL.append(recall)


def get_data(filename, balanced=False):
    df = pd.read_csv(filename)
    texts = df['texts'].tolist()
    labels = df['labels'].tolist()

    # test balanced
    '''
    if balanced:
        new_texts = list(texts)
        new_labels = list(labels)
        for text, label in zip(texts, labels):
            if 'B-SHOOTER' in label:
                new_texts += [text] * 9
                new_labels += [label] * 9
        return new_texts, new_labels'''


    return texts, labels


if __name__ == '__main__':
    args = _handle_arguments()

    train_X, train_Y = get_data('victim/new_train.csv', True)
    dev_X, dev_Y = get_data('victim/new_dev.csv', True)
    #train_X += dev_X
    #train_Y += dev_Y
    model, tokenizer = train(train_X, train_Y, args.lr, args.cuda_available, args.epochs, args.model, args.is_balance, args.batch_size, args.max_seq_length)
    #model = torch.load('output/model')
    #tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased') # cased!


    test_X, test_Y = get_data('victim/new_test.csv')
    eval_results = evaluate(model, test_X, test_Y, tokenizer, args.cuda_available, args.batch_size, args.max_seq_length, args.model, args.lr, args.epochs)


    # X, Y = get_data('data.csv')

    # kf = KFold(n_splits=5)
    # for i, (train_index, test_index) in enumerate(kf.split(X)):
    #     print('Fold', i)
    #     train_X = [X[i] for i in train_index]
    #     train_Y = [Y[i] for i in train_index]

    #     test_X = [X[i] for i in test_index]
    #     test_Y = [Y[i] for i in test_index]

    #     model, tokenizer = train(train_X, train_Y, args.lr, args.cuda_available, args.epochs, args.model, args.is_balance, args.batch_size, args.max_seq_length)

    #     # model = torch.load('output/model')
    #     # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased') # cased!
    #     eval_results = evaluate(model, test_X, test_Y, tokenizer, args.cuda_available, args.batch_size, args.max_seq_length)

    # with open('./data/{}_{}_{}_{}_{}.txt'.format(args.model, args.lr, args.epochs, args.batch_size, args.max_seq_length), 'w') as wf:
    #     if len(ACCURACY) != 0: wf.write('Avg accuracy: {}\n'.format(sum(ACCURACY)/len(ACCURACY)))
    #     if len(PRECISION) != 0: wf.write('Avg precision: {}\n'.format(sum(PRECISION)/len(PRECISION)))
    #     if len(RECALL) != 0: wf.write('Avg recall: {}\n'.format(sum(RECALL)/len(RECALL)))
    #     if (sum(PRECISION)/len(PRECISION) + sum(RECALL)/len(RECALL)) != 0: wf.write('F1: {}'.format(2 * sum(PRECISION)/len(PRECISION) * sum(RECALL)/len(RECALL) / (sum(PRECISION)/len(PRECISION) + sum(RECALL)/len(RECALL))))
