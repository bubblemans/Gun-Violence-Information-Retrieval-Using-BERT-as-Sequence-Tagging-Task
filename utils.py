import argparse
import pandas as pd


TRANSFORMER_PATH = 'huggingface/pytorch-transformers'

LABEL_MAPPING = {
    'B': 0,
    'I': 1,
    'O': 2
}


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='victim', type=str, required=False, help='Input data directory that contains train.csv and eval.csv')
    parser.add_argument('--output_dir', default='victim/output', type=str, required=False, help='Output data directory')
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help='Learning rate')
    parser.add_argument('--cuda_available', default=True, type=bool, required=False, help='Decide whether to use GPU')
    parser.add_argument('--epochs', default=1, type=int, required=False, help='Number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='Number of batches')
    parser.add_argument('--max_seq_length', default=256, type=int, required=False, help='Number of max sequence length')
    parser.add_argument('--model_type', default='Linear', type=str, required=False, help='Linear, LSTM, BiLSTM')
    parser.add_argument('--model', default='', type=str, required=False, help='path to model')
    parser.add_argument('--is_balance', default=True, type=bool, required=False, help='choose to use balance data or unbalanced data')
    parser.add_argument('--patience', default=10, type=int, required=False, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--min_delta', default=0, type=float, required=False, help='Minimum change in the monitored quantity to qualify as an improvement')
    parser.add_argument('--baseline', default=0.0001, type=float, required=False, help='Training will stop if the model doesn\'t show improvement over the baseline')
    return parser.parse_args()


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
                    label = LABEL_MAPPING[labels_org[i - 1 - token_bias_num][0]]
                    labels.append(label)
                else:
                    labels.append(1)  # 1 is I
                token_bias_num += 1
            else:
                word_num += 1
                label = LABEL_MAPPING[labels_org[i - token_bias_num][0]]
                labels.append(label)

        # manually pad tokens and labels if their lengths are over the max sequence length
        if len(tokens) < max_seq_length:
            tokens += [''] * (max_seq_length - len(tokens))
            labels += [LABEL_MAPPING['O']] * (max_seq_length - len(labels))

        token_batch.append(tokens)
        label_batch.append(labels)

    return token_batch, label_batch

# loading data from input file
def get_data(filename, balanced=False):
    df = pd.read_csv(filename)
    texts = df['texts'].tolist()
    labels = df['labels'].tolist()

    if balanced:
        new_texts = list(texts)
        new_labels = list(labels)
        for text, label in zip(texts, labels):
            if 'B' in label:
                new_texts += [text] * 9
                new_labels += [label] * 9
        return new_texts, new_labels

    return texts, labels