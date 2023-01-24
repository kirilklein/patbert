import numpy as np
import torch
from torch import nn

from patbert.common import common, medical
from patbert.features import utils

class Embedding(nn.Embedding):
    """Modifying the embedding module to handle integer input as well"""
    def forward(self, input):
        if isinstance(input, int):
            # handle integer input
            return self.weight[input]
        else:
            # handle tensor input
            return super(Embedding, self).forward(input)

class StaticHierarchicalEmbedding(nn.Module):
    def __init__(self, data, cfg):
        super(StaticHierarchicalEmbedding, self).__init__()
        """
        kappa: exponent to make vectors shorter with each hierarchy level
        alpha: vectors at level 0 are scaled by alpha after being normalized"""
        # TODO: make fully trainable scaling
        _, vocab, self.int2int = data
        self.embedding_dim = cfg.model.hidden_size
        self.num_levels = cfg.model.embedding.num_levels
        self.sks = medical.SKSVocabConstructor(vocab, num_levels=self.num_levels)
        self.vocabs = self.sks()
    
        self.fully_trainable_scaling = cfg.model.embedding.fully_trainable_scaling

        if self.fully_trainable_scaling:
            self.kappa = cfg.model.embedding.kappa
            self.alpha = cfg.model.embedding.alpha
        else:
            self.kappa = torch.tensor(float(cfg.model.embedding.kappa), 
                requires_grad=cfg.model.embedding.kappa_trainable)
            self.alpha = torch.tensor(float(cfg.model.embedding.alpha), 
                requires_grad=cfg.model.embedding.alpha_trainable)

        self.embedding_ls = self.initialize_static_embeddings()
        self.set_zero_weight()
        self.set_to_static()

    def forward(self, ids, values=None):
        """Outputs a tensor of shape levels x len x emb_dim"""
        if values is None:
            values = torch.ones_like(ids)
        if ids.shape[1]!=values.shape[1]:
            raise ValueError("Codes and values must have the same length")
        #assert False
        self.id_arr_ls = []
        for dic in self.int2int: # we get one batch of ids for each level
            self.id_arr_ls.append(utils.remap_values(dic, ids))
        self.embedding_tsr = self.get_embedding_tsr() # levels x batch x len x emb_dim
        assert False
        self.scale_embedding_tsr()
        self.multiply_embedding_tsr_by_values(values)
        self.embedding_tsr = torch.sum(self.embedding_tsr, dim=0) # sum over levels dim
        return self.embedding_tsr

    def initialize_static_embeddings(self):
        embedding_ls = [] # TODO: think about vectorizing
        for vocab in self.vocabs:
            embedding_ls.append(nn.Embedding(len(vocab), self.embedding_dim))
        return embedding_ls

    def multiply_embedding_tsr_by_values(self, values):
        """Multiply embedding tensor by value at the right hierarchy level"""
        value_tsr = self.get_value_tsr(self.id_arr_ls, values)
        self.embedding_tsr *= value_tsr.unsqueeze(-1)

    def scale_embedding_tsr(self):
        """Scale embedding_mat to have decreasing length with each level of hierarchy"""
        # shorter vectors at lower levels of hierarchy
        level_mult = 1/(torch.arange(1, self.num_levels+1)**self.kappa) 
        if self.fully_trainable_scaling:
            level_mult.requires_grad = True
        self.embedding_tsr = level_mult.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*self.embedding_tsr
    
    def get_embedding_tsr(self):
        """Returns a tensor of shape levels x batch x len x emb_dim"""
        arr_ls = []
        # TODO: we should get rid of this for loop
        for id_arr, embedding in zip(self.id_arr_ls, self.embedding_ls): # loop through levels
            arr = embedding(torch.LongTensor(id_arr))
            lens = torch.norm(arr, dim=-1).unsqueeze(-1).expand_as(arr) # last dimension is emb_dim
            # new_arr = torch.copy(arr, fill_value=float('nan'))
            mask = (lens != 0) # avoid division by zero
            arr[mask] = arr[mask] / lens[mask]
            arr = arr * self.alpha
            arr_ls.append(arr) 
        # concatenate arr_ls to get 4d tensor
        embedding_tsr = torch.stack(arr_ls) 
        return embedding_tsr

    def set_to_static(self):
        for embedding in self.embedding_ls:
            embedding.weight.requires_grad = False

    @staticmethod
    def get_value_tsr(id_arr_ls, values):
        """Store values in a tensor of shape id_tsr"""
        id_tsr = torch.from_numpy(np.stack(id_arr_ls)) 
        value_tsr = torch.ones_like(id_tsr).to(torch.float64)
        values = values.to(value_tsr.dtype)
        last_non_zero = common.get_last_nonzero_idx(id_tsr,0)
        value_tsr = value_tsr.scatter(0, last_non_zero.unsqueeze(0), values.unsqueeze(0))
        return value_tsr

    def set_zero_weight(self):
        """Initialize first index to be a zero vector"""
        for embedding in self.embedding_ls:
            embedding.weight.data[0] = torch.zeros(embedding.weight.data[0].shape)


class VisitEmbedding(Embedding):
    def __init__(self, embedding_dim:int, max_num_visits=500):
        """
        For visit embeddings, we use standard vocabulary"""
        super(VisitEmbedding, self).__init__(max_num_visits, embedding_dim)
        # TODO: make fully trainable scaling
        self.embedding_dim = embedding_dim
        self.max_num_visits = max_num_visits
        self.embedding = Embedding(max_num_visits, embedding_dim)
        self.set_zero_weight()
    def set_zero_weight(self):
        """Initialize first index to be a zero vector"""
        self.embedding.weight.data[0] = torch.zeros(self.embedding.weight.data[0].shape)
    def forward(self, visits):
        return self.embedding(visits)

class Time2Vec(nn.Module):
    """Time2Vec embedding, check code, use now alternative instead"""
    def __init__(self,  out_features, in_features=1, activation='sin'):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))

        if activation=='sin':
            self.f = torch.sin
        elif activation=='cos':
            self.f = torch.cos
        else:
            raise ValueError('activation must be sin or cos')

    def forward(self, tau):
        return self.t2v(tau)
    
    def t2v(self, tau, arg=None):
        print(tau)
        print(torch.sum(tau))
        print("tau dtype", tau.dtype)
        print("w dtype", self.w.dtype)
        print("b dtype", self.b.dtype)
        if arg:
            v1 = self.f(torch.matmul(tau, self.w) + self.b, arg)
        else:
            v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], 1)

def get_positional_embeddings(cfg):
    """Returns a dictionary of positional embeddings for each channel
    Use time2vec for absolute position and age and VisitEmbedding for visits"""
    pos_embeddings = {}
    for c in cfg.data.channels:
        if c in ['abs_pos', 'ages']:
            pos_embeddings[c] = Time2Vec(cfg.model.hidden_size, in_features=cfg.data.pad_len) #check this
        elif c=='visits':
            pos_embeddings[c] = VisitEmbedding(cfg.model.hidden_size)
        elif c == 'values':
            pass
        else:
            raise ValueError(f"Channel {c} not supported")

    return pos_embeddings

def get_add_params(channels, epsilon=1e-5):
    add_params = {}
    for c in channels:
        add_params[c] = nn.parameter.Parameter(torch.tensor(epsilon), requires_grad=True)
    return add_params