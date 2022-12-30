import random
import numpy as np
from numpy.random import default_rng
rng = default_rng()

#TODO dont mask cls and sep tokens
def random_mask(codes, vocab, mask_prob=0.15):
    """mask code with 15% probability, 80% of the time replace with [MASK], 
        10% of the time replace with random token, 10% of the time keep original"""
    masked_codes = codes
    labels = len(codes) * [-100] # -100 is ignored by loss function
    prob = rng.uniform()
    for i, _ in enumerate(masked_codes):
        if prob<mask_prob:
            prob = rng.uniform()   
            if prob < 0.8:
                masked_codes[i] = vocab['MASK']
                labels[i] = vocab['UNK']
            # 10% randomly change token to random token
            elif prob < 0.9:
                masked_codes[i] = rng.choice(list(vocab.values())[9:]) # first 9 tokens are special!
    return masked_codes, labels

def seq_padding(seq, max_len, vocab):
    return seq + (max_len-len(seq)) * [vocab['PAD']]

#TODO torch.utils.data.random_split