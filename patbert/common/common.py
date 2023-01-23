import os
from os.path import dirname, join, realpath

import pickle as pkl
import numpy as np
import torch


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

def load_tokenized_data(data_name, vocab_only=False):
    """Loads data and vocab from data folder"""
    base_dir = dirname(dirname(dirname(realpath(__file__))))
    data_dir = join(base_dir, 'data')
    vocab = torch.load(join(data_dir, 'vocabs', data_name + '.pt'))
    if vocab_only:
        return vocab
    data = torch.load(join(data_dir, 'tokenized', data_name + '.pt'))
    if os.path.exists(join(data_dir, 'hierarchy_vocabs', data_name +'.pt')):
        int2int = torch.load(join(data_dir, 'hierarchy_vocabs', \
            data_name + '.pt'))
        return data, vocab, int2int
    return data, vocab

def load_processed_data(data_name):
    """Loads data from data folder""" 
    base_dir = dirname(dirname(dirname(realpath(__file__))))
    data_dir = join(base_dir, 'data', 'processed')
    try:
        with open(join(data_dir, 'processed' , data_name + '.pkl'), 'rb')as f:
            return pkl.load(f)
    except:
        try:
            return torch.load(join(data_dir, 'processed', data_name + '.pt'))
        except:
            raise ValueError(f"Could not find {data_name} in {data_dir}")
            
    