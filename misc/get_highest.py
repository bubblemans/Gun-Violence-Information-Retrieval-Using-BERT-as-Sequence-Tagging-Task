import os 


def get_highest(target='shooter', model='Linear'):
    directory = './{}/output/'.format(target)
    files = os.listdir(directory)
    max_f1 = 0
    max_f1_setup = None
    for f in files:
        if f.startswith(model):
            with open(directory + f) as rf:
                lines = rf.readlines()
                f1_str = lines[-1].replace('F1: ', '').replace('\n', '')
                f1 = float(f1_str) if f1_str else 0
                if max_f1 < f1:
                    max_f1 = f1
                    max_f1_setup = f

    print('model:', max_f1_setup, 'f1:', max_f1)

if __name__ == '__main__':
    for target in ['shooter', 'victim']:
        print('{}:'.format(target))
        for model in ['Linear', 'LSTM', 'BiLSTM', 'crf_Linear', 'crf_LSTM', 'crf_BiLSTM']:
            get_highest(target, model)