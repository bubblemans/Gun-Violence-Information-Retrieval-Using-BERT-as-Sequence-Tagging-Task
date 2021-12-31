import torch
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation


text = "Demarcus"
# text = "Dominique Adams"
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
tokenized_text = [t[0] for t in pre_tokenizer.pre_tokenize_str(text)]
print(tokenized_text)

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
tokens = tokenizer.tokenize(text, truncation=True, max_length=128)
print(tokens)