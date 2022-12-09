from patbert.features.utils import random_mask

def test_random_mask():
    codes = [1,2,3,4,5,6,7,8,9,10]
    vocab = {'CLS':0, 'PAD':1, 'SEP':2, 'MASK':3, 'UNK':4, 'A':5, 'B':6, 'C':7, 'D':8, 'E':9, 'F':10}
    masked_codes, labels = random_mask(codes, vocab, mask_prob=0.15)   
    assert len(masked_codes)==len(codes)
    assert len(labels)==len(codes)
    
