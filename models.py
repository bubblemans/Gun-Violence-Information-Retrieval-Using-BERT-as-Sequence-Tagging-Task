import torch
import torch.nn as nn

from torchcrf import CRF


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


class BERT_CRF_Linear(nn.Module):
    def __init__(self, num_labels):
        super(BERT_CRF_Linear, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        config.max_position_embeddings = 1024
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.classifier = nn.Linear(768, num_labels)
        self.CRF_model = CRF(num_labels, batch_first=True)

    def forward(self, tokens_tensor, segments_tensors, labels=None):
        bert_output = self.bert(tokens_tensor, token_type_ids=segments_tensors)
        last_hidden_state = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output

        logits = self.classifier(last_hidden_state)

        # the CRF layer of NER labels
        crf_loss_list = self.CRF_model(logits, labels)
        #crf_loss_list = self.CRF_model(last_hidden_state, labels)

        crf_loss = torch.mean(-crf_loss_list)
        crf_predict = self.CRF_model.decode(logits)

		# the classifier of category & polarity
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.permute(0, 2, 1), labels)
        return torch.tensor(crf_predict).to('cuda'), logits, loss


class BERT_CRF_LSTM(nn.Module):
    def __init__(self, num_labels):
        super(BERT_CRF_LSTM, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        config.max_position_embeddings = 1024
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.lstm = nn.LSTM(768, 768)
        self.classifier = nn.Linear(768, num_labels)
        self.CRF_model = CRF(num_labels, batch_first=True)

    def forward(self, tokens_tensor, segments_tensors, labels=None):
        bert_output = self.bert(tokens_tensor, token_type_ids=segments_tensors)
        last_hidden_state = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output

        lstm_out, _ = self.lstm(last_hidden_state)
        #logits, _ = self.lstm(last_hidden_state)

        logits = self.classifier(lstm_out)

        # the CRF layer of NER labels
        crf_loss_list = self.CRF_model(logits, labels)
        crf_loss = torch.mean(-crf_loss_list)
        crf_predict = self.CRF_model.decode(logits)

		# the classifier of category & polarity
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.permute(0, 2, 1), labels)
        return torch.tensor(crf_predict).to('cuda'), logits, loss


class BERT_CRF_BiLSTM(nn.Module):
    def __init__(self, num_labels):
        super(BERT_CRF_BiLSTM, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        config.max_position_embeddings = 1024
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.lstm = nn.LSTM(768, 768, bidirectional=True)
        self.classifier = nn.Linear(768, num_labels)
        # self.classifier = nn.Linear(768 * 2, num_labels)
        self.CRF_model = CRF(num_labels, batch_first=True)

    def forward(self, tokens_tensor, segments_tensors, labels=None):
        bert_output = self.bert(tokens_tensor, token_type_ids=segments_tensors)
        last_hidden_state = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output

        lstm_out, _ = self.lstm(last_hidden_state)
        lstm_out = lstm_out[:, :, :768] + lstm_out[:, :, 768:]

        logits = self.classifier(lstm_out)

        # the CRF layer of NER labels
        crf_loss_list = self.CRF_model(logits, labels)
        #crf_loss_list = self.CRF_model(lstm_out, labels)
        crf_loss = torch.mean(-crf_loss_list)
        crf_predict = self.CRF_model.decode(logits)

		# the classifier of category & polarity
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.permute(0, 2, 1), labels)
        return torch.tensor(crf_predict).to('cuda'), logits, loss