import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM

from patbert.features import embeddings
from patbert.features.embeddings import StaticHierarchicalEmbedding

class MetaClass(type):
    @staticmethod
    def wrap(forward):
        """Wraps forward method to add additional tasks of every child class"""
        def new_forward(self, batch):
            outputs = forward(self, batch)
            if self.cfg.training.tasks.prolonged_los_global:
                outputs = self.get_prolonged_los(self, outputs, batch)
            return outputs
        return new_forward
    def __new__(cls, name, bases, attrs):
        """If the class has a 'run' method, wrap it"""
        if 'forward' in attrs:
            attrs['forward'] = cls.wrap(attrs['forward'])
        return super(MetaClass, cls).__new__(cls, name, bases, attrs)

class MedBERT(torch.nn.Module):
    def __init__(self, data=None, cfg=None) -> None:
        __metaclass__ = MetaClass
        super(MedBERT, self).__init__()
        self.cfg = cfg
        _, vocab, _ = data
        self.model_config = BertConfig(vocab_size=len(vocab), **cfg.model.architecture)
        self.main_embedding = nn.Embedding(len(vocab), cfg.model.architecture.hidden_size)
        self.bert = BertForMaskedLM(self.model_config)
        if cfg.training.tasks.prolonged_los_global:
            self.head_nn = nn.Linear(cfg.model.architecture.hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        self.plos_head = nn.Linear(cfg.model.architecture.hidden_size, 1)

    def forward(self, batch):
        outputs = self.bert(
            inputs_embeds=batch['idx'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        return outputs

    def parameters(self):
        return self.bert.parameters()
    
    def get_prolonged_los(self, outputs, batch):
        prediction = self.sigmoid(self.head_nn(outputs['last_hidden_state'])) # TODO: check if this produces the right shape
        outputs['loss'] += torch.nn.functional.binary_cross_entropy(prediction, batch['prolonged_los'])
        return outputs


class StaticHierarchicalBERT(torch.nn.Module):
    def __init__(self, data=None, cfg=None) -> None:
        super(StaticHierarchicalBERT, self).__init__()
        self.cfg = cfg
        _, vocab, _ = data
        self.model_config = BertConfig(vocab_size=len(vocab), **cfg.model.architecture)
        # embeddings
        self.main_embedding = StaticHierarchicalEmbedding(data, cfg)
        self.pos_embeddings = embeddings.get_positional_embeddings(cfg)
        self.add_params = embeddings.get_add_params(cfg.data.channels)
        hidden_size = cfg.model.architecture.hidden_size
        self.fc0 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.bert = BertForMaskedLM(self.model_config)

    def forward(self, batch):
        X = self.main_embedding(batch['idx'], batch['values'])
        for c in self.cfg.data.channels:
            if c != 'values':
                X += self.add_params[c] * self.pos_embeddings[c](batch[c])
        X = self.gelu(self.fc0(X))
        outputs = self.bert(
            inputs_embeds=X,
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        return outputs

    def parameters(self):
        return self.bert.parameters()
