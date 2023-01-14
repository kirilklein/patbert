from patbert.features import tokenizer
import pickle
import os
from os.path import join, dirname, realpath


base_dir = dirname(dirname(dirname(realpath(__file__))))
data_dir = join(base_dir, 'data')

def test_tokenizer(max_len=20, len_background=4):
    assert os.path.exists(join(data_dir, 'raw', 'synthetic.pkl')), "Generate example data first" 
    with open(join(data_dir, 'raw', 'synthetic.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    Tokenizer = tokenizer.HierarchicalTokenizer(max_len=max_len)
    tok_data = Tokenizer.batch_encode(data)
    assert len(data)==len(tok_data), "Number of patients should be the same"
    for seq in tok_data:
        assert len(seq['codes']) == len(seq['visits'])  == len(seq['idx'])\
             == len(seq['ages']) == len(seq['los']) == len(seq['abs_pos']) == len(seq['values']), "All lists should have the same length"
        assert len(seq['codes']) <= max_len - len_background, "Sequence should be truncated"