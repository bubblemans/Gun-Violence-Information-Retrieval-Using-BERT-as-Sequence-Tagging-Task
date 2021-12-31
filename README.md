# Gun-Violence-Information-Retrieval-Using-BERT-as-Sequence-Tagging-Task
Code for "Gun Violence News Information Retrieval using BERT as Sequence Tagging Task" (IEEE BigData 2021)
## Environment
Besides the packages in the `requirements.txt`, you need `apex`. Please use the following commands.
```
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-10.1
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

# Usage
There are two steps:
1. Structure directories 
2. Preprocess
3. Train and evaluate


## Structure input/output directories
```
mkdir shooter
mkdir shooter/output
mkdir victim
mkdir victim/output
```
## Preprocess from dataset

```python
python process.py --target_type victim
python process.py --target_type shooter
```

## Train and evaluate
We do not separate training and evaluating, so if you run the script, you will train and evaluate models at the same time. You can choose whether you want to read from `victim` or `shooter`. You can set `model_type` to `Linear`, `LSTM`, or `BiLSTM`.
```
python train.py \
    --input_dir victim \
    --output_dir victim/output \
    --lr 1e-4 --cuda_available True \
    --epochs 1 --batch_size 1 \
    --max_seq_length 256 \
    --model_type Linear \
    --is_balance True \
    --patience 10 \
    --min_delta 0 \
    --baseline 0.0001
```

```
python train_crf.py \
    --input_dir victim \
    --output_dir victim/output \
    --lr 1e-4 --cuda_available True \
    --epochs 1 --batch_size 1 \
    --max_seq_length 256 \
    --model_type Linear \
    --is_balance True \
    --patience 10 \
    --min_delta 0 \
    --baseline 0.0001
```