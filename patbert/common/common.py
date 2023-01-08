import os
import numpy as np


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