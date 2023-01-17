import os
import pickle
from os.path import dirname, join, realpath

from patbert.features import tokenizer

base_dir = dirname(dirname(dirname(realpath(__file__))))
data_dir = join(base_dir, 'data')

def test_tokenizer(max_len=20):
    assert os.path.exists(join(data_dir, 'raw', 'synthetic.pkl')), "Generate example data first" 
    with open(join(data_dir, 'raw', 'synthetic.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    Tokenizer = tokenizer.EHRTokenizer(max_len=max_len)
    tok_data = Tokenizer.batch_encode(data)
    assert len(data)==len(tok_data), "Number of patients should be the same"
    for seq in tok_data:
        len_ = len(seq['codes'])
        for key in ['visits', 'idx', 'ages', 'los', 'abs_pos', 'values']:
            assert len_ == len(seq[key]), f"sequence {key} has a different length of {len(seq[key])}!={len_}"
        assert len(seq['codes']) <= max_len, "Sequence should be truncated"