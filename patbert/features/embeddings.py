import string
from operator import itemgetter

import numpy as np
import torch
from torch import nn

from patbert.common import common, medical


class StaticHierarchicalEmbedding(nn.Embedding):
    def __init__(self, vocab, embedding_dim:int, num_levels:int=6, kappa:int=3, alpha:int=20,
            alpha_trainable:bool=False, kappa_trainable:bool=False, fully_trainable_scaling=False):
        """
        kappa: exponent to make vectors shorter with each hierarchy level
        alpha: vectors at level 0 are scaled by alpha after being normalized"""
        # TODO: make fully trainable scaling
        self.embedding_dim = embedding_dim
        self.sks = medical.SKSVocabConstructor(num_levels=num_levels)
        self.vocabs = self.sks()
        self.num_levels = num_levels
        self.fully_trainable_scaling = fully_trainable_scaling
        self.main_inv_vocab = {v:k for k,v in vocab.items()}
        if self.fully_trainable_scaling:
            self.kappa = kappa
            self.alpha = alpha
        else:
            self.kappa = torch.tensor(float(kappa), requires_grad=kappa_trainable)
            self.alpha = torch.tensor(float(alpha), requires_grad=alpha_trainable)

    def __call__(self, ids, values=None):
        """Outputs a tensor of shape levels x len x emb_dim"""
        if values is None:
            values = torch.ones_like(ids)
        if len(ids)!=len(values):
            raise ValueError("Codes and values must have the same length")
        ids = ids.tolist()
        codes = itemgetter(*ids)(self.main_inv_vocab)
        self.id_arr_ls = self.get_ids_from_codes(codes)
        self.embedding_ls = self.initialize_static_embeddings()
        self.set_zero_weight()
        self.set_to_static()
        self.embedding_mat = self.get_embedding_mat()
        self.scale_embedding_mat()
        self.multiply_embedding_mat_by_values(values)
        return self.embedding_mat

    def initialize_static_embeddings(self):
        embedding_ls = [] # TODO: think about vectorizing
        for vocab in self.vocabs:
            embedding_ls.append(Embedding(len(vocab), self.embedding_dim))
        return embedding_ls

    def multiply_embedding_mat_by_values(self, values):
        """Multiply embedding_mat by values"""
        value_mat = self.get_value_mat(self.id_arr_ls, values)
        self.embedding_mat *= value_mat.unsqueeze(-1)

    def scale_embedding_mat(self):
        """Scale embedding_mat to have decreasing length with each level of hierarchy"""
        # shorter vectors at lower levels of hierarchy
        level_mult = 1/(torch.arange(1, self.num_levels+1)**self.kappa) 
        if self.fully_trainable_scaling:
            level_mult.requires_grad = True
        self.embedding_mat = level_mult.unsqueeze(-1).unsqueeze(-1)*self.embedding_mat
        
    def get_embedding_mat(self):
        """Returns a tensor of shape levels x len x emb_dim"""
        arr_ls = []
        for id_arr, embedding in zip(self.id_arr_ls, self.embedding_ls):
            arr = embedding(torch.LongTensor(id_arr))
            lens = torch.norm(arr, dim=0)
            arr  = (arr * self.alpha)/lens          
            arr_ls.append(arr) 
        # concatenate arr_ls to get 3d tensor
        embedding_mat = torch.stack(arr_ls) 
        return embedding_mat

    def set_to_static(self):
        for embedding in self.embedding_ls:
            embedding.weight.requires_grad = False

    def get_ids_from_codes(self, codes):
        id_arr_ls = [] # TODO: think about vectorizing
        codes = np.array(codes, dtype='str')
        for vocab in self.vocabs:
            id_arr_ls.append(np.vectorize(lambda x: vocab.get(x, 0))(codes))
        return id_arr_ls

    @staticmethod
    def get_value_mat(id_arr_ls, values):
        if isinstance(values, type(list)):
            values = torch.tensor(values)
        print(type(values))
        id_arr = torch.from_numpy(np.stack(id_arr_ls))
        value_mat = torch.ones_like(id_arr).to(torch.float64)
        last_non_zero = common.get_last_nonzero_idx(id_arr,0)
        ids1 = torch.arange(len(last_non_zero))
        value_mat[last_non_zero, ids1] = values.to(value_mat.dtype)
        return value_mat

    def set_zero_weight(self):
        """Initialize first index to be a zero vector"""
        for embedding in self.embedding_ls:
            embedding.weight.data[0] = torch.zeros(embedding.weight.data[0].shape)

