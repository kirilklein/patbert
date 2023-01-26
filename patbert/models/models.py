import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM

from patbert.features import embeddings
from patbert.features.embeddings import StaticHierarchicalEmbedding

class MedBERT(torch.nn.Module):
    def __init__(self, data=None, cfg=None) -> None:
        super(MedBERT, self).__init__()
        self.cfg = cfg
        _, vocab, _ = data
        self.model_config = BertConfig(vocab_size=len(vocab), **cfg.model)
        self.main_embedding = nn.Embedding(len(vocab), cfg.model.hidden_size)
        self.bert = BertForMaskedLM(self.model_config)

    def forward(self, batch):
        return self.bert(inputs_embeds=batch['idx'], attention_mask=batch['attention_mask'], labels=batch['labels'])

    def parameters(self):
        return self.bert.parameters()


class StaticHierarchicalBERT(torch.nn.Module):
    def __init__(self, data=None, cfg=None) -> None:
        super(StaticHierarchicalBERT, self).__init__()
        self.cfg = cfg
        _, vocab, _ = data
        self.model_config = BertConfig(vocab_size=len(vocab), **cfg.model)
        # embeddings
        self.main_embedding = StaticHierarchicalEmbedding(data, cfg)
        self.pos_embeddings = embeddings.get_positional_embeddings(cfg)
        self.add_params = embeddings.get_add_params(cfg.data.channels)
        self.fc0 = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)
        self.gelu = nn.GELU()
        self.bert = BertForMaskedLM(self.model_config)

    def forward(self, batch):
        X = self.main_embedding(batch['idx'], batch['values']) 
        # print(X)
        for c in self.cfg.data.channels:
            if c!='values':
                X += self.add_params[c] * self.pos_embeddings[c](batch[c])
        X = self.gelu(self.fc0(X))
        outputs = self.bert(inputs_embeds=X, attention_mask=batch['attention_mask'], labels=batch['labels'])
        return outputs

    def parameters(self):
        return self.bert.parameters()