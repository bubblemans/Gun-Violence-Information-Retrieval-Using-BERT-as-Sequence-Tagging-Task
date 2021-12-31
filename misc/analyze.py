import pandas as pd
import json
import argparse


def _handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/train.tsv', type=str, required=False, help='Input data file that is train.tsv, dev.tsv, or test.tsv')
    parser.add_argument('--target_type', default='shooter', type=str, required=False, help='victim or shooter')
    return parser.parse_args()


def get_distribution(input_file, target_type):
    df = pd.read_csv(input_file, sep='\t')
    jsons = df['Json'].tolist()

    num_of_no_target = 0
    for data in jsons:
        data = json.loads(data)
        if target_type + '-section' not in data:
            num_of_no_target += 1
        elif len(data[target_type + '-section']) == 0:
            num_of_no_target += 1
        else:
            target = data[target_type + '-section'][0]['name']['value']
    
    print('From:', input_file)
    print('Total:', len(jsons))
    print('Num of no {}: {}'.format(target_type, num_of_no_target))
    print('Num of {}: {}'.format(target_type, len(jsons) - num_of_no_target))



if __name__ == '__main__':
    args = _handle_arguments()
    input_file = args.input_file
    target_type = args.target_type
    get_distribution(input_file, target_type)