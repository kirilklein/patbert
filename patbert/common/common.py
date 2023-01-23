import os
from os.path import dirname, join, realpath

import pickle as pkl
import numpy as np
import torch

from patbert.features import utils

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_encodings_from_npz(file):
    """Returns patient keys and encodings from a npz file"""
    npzfile = np.load(file, allow_pickle=True)
    return npzfile['pat_ids'], npzfile['pat_vecs']

def get_inverse_dic(vocab_dic):
    """Returns the inverse of a dictionary"""
    return {v: k for k, v in vocab_dic.items()}

def check_same_elements(lst):
    return all(x == lst[0] for x in lst)

def check_unique(lst):
    return len(lst) == len(set(lst))

def inspect_dic(dic, start_str='', end_str=''):
    """Return dic where keys start with start_str, end/or ends with end_str"""
    return {k:v for k, v in dic.items() if k.startswith(start_str) and k.endswith(end_str)}

def key_length(dic, length):
    """Return part of dictionary where keys have a certain length"""
    return {k:v for k,v in dic.items() if len(k)==length}

def get_last_nonzero_idx(x, axis):
    """Returns index of the first 0 along axis or -1 if no zero"""
    mask = (x == 0).to(int)
    last_nonzero = torch.argmax(mask, dim=axis)-1
    any_mask = torch.any(mask, dim=axis)
    last_nonzero[~any_mask] = x.shape[axis]-1 # last element along this axis
    return last_nonzero


def load_data(cfg):
    return Data(cfg).get_tokenized_data()

class Data:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_name = cfg.data.name
        self.data_dir = self.get_data_dir()   
            
    def get_tokenized_data(self, vocab_only=False):
        """Loads or creates tokenized data"""
        if self.check_tokenized_data_exists():
            return self.load_tokenized_data(vocab_only)
        else:
            return utils.create_tokenized_data(self.cfg)
        
    def load_tokenized_data(self, vocab_only):
        tok_dir = join(self.data_dir, 'tokenized')
        vocab = torch.load(join(tok_dir, self.data_name +  '_vocab.pt'))
        if vocab_only:
            return vocab
        data = torch.load(join(tok_dir, self.data_name + '.pt'))
        if os.path.exists(join(tok_dir,  self.data_name +'_hierarchy_mapping.pt')):
            int2int = torch.load(join(tok_dir, \
                self.data_name +'_hierarchy_mapping.pt'))
            return data, vocab, int2int
        return data, vocab

    def load_processed_data(self):
        """Loads processed data from data_dir, either default or specified in config""" 
        try:
            with open(join(self.data_dir, 'processed', self.data_name + '.pkl'), 'rb')as f:
                return pkl.load(f)
        except:
            try:
                return torch.load(join(self.data_dir, 'processed', self.data_name + '.pt'))
            except:
                raise ValueError(f"Could not find {self.data_name} in {self.data_dir}")
    
    def get_data_dir(self):
        """Returns data directory"""
        if isinstance(self.cfg.data.dir, type(None)):
            base_dir = dirname(dirname(dirname(realpath(__file__))))
            data_dir = join(base_dir, 'data')
        else:
            data_dir = self.cfg.data.dir
        return data_dir

    def check_tokenized_data_exists(self):
        """Checks if tokenized data exists, for hierarchical _hierarchy_mapping needed"""
        tok_dir = join(self.data_dir, 'tokenized')
        tok_files = os.listdir(tok_dir)
        if self.cfg.model.embedding.hierarchical:
            if self.data_name +  '_vocab.pt' in tok_files \
                and self.data_name + '.pt' in tok_files\
                    and self.data_name +'_hierarchy_mapping.pt' in tok_files:
                return True
            else:
                return False
        else:
            if self.data_name +  '_vocab.pt' in tok_files \
                and self.data_name + '.pt' in tok_files:
                return True
            else:
                return False