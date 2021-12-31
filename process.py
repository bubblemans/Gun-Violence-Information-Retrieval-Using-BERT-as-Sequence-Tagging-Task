import argparse
import pandas as pd
import json

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation


def _handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_type', default='victim', type=str, required=False, help='Input data directory that contains benchmark Data')
    return parser.parse_args()


def _gen_label(words, target):
    target_len = len(target)
    tags = ['O'] * len(words)
    for i in range(0, len(words)):
        try:
            if ' '.join(words[i:i+target_len]) == ' '.join(target):
                tags[i] = 'B'
                for j in range(i+1, i+target_len):
                    tags[j] = 'I'
        except IndexError as e:
            print(e)
            exit()

    return ' '.join(tags)

def preprocess(input_file, output_file, target_type):
    df = pd.read_csv(input_file, sep='\t')
    texts = df['Full text'].tolist()
    jsons = df['Json'].tolist()

    new_texts = []
    labels = []

    for text, data in zip(texts, jsons):
        try:
            # use BERT tokenizer to process whitespace and punctuaction
            data = json.loads(data)
            text = text.replace('\u200b', '')
            pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
            tokenized_text = [t[0] for t in pre_tokenizer.pre_tokenize_str(text)]

            target = data[target_type + '-section'][0]['name']['value']
            tokenized_target = [t[0] for t in pre_tokenizer.pre_tokenize_str(target)]

            # if no target or empty in array, mark every token as O
            if not target:
                raise IndexError

            # generate labels for each tokenized token
            label = _gen_label(tokenized_text, tokenized_target)

            # keep tokenized text that has less than 512 text-length
            if label and len(tokenized_text) < 512:
                new_texts.append(' '.join(tokenized_text))
                labels.append(label)

        except IndexError:
            # mark every token as O
            new_texts.append(' '.join(tokenized_text))
            label = ' '.join(['O'] * len(tokenized_text))
            labels.append(label)

    data = list(zip(new_texts, labels))
    df = pd.DataFrame(data)
    df.columns = ['texts', 'labels']
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    args = _handle_arguments()
    target = args.target_type

    if target not in ['victim', 'shooter']:
        print('target_type must be victim or shooter')
        exit()

    preprocess('data/train.tsv', target + '/train.csv', target)
    preprocess('data/dev.tsv', target + '/dev.csv', target)
    preprocess('data/test.tsv', target + '/test.csv', target)