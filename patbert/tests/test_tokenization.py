from patbert.features import tokenizer
import pickle
import os

def test_tokenizer(max_len=20, len_background=4):
    assert os.path.exists("data\\raw\\simulated.pkl"), "Generate example data first" 
    with open("data\\raw\\simulated.pkl", 'rb') as f:
        data = pickle.load(f)
    
    Tokenizer = tokenizer.HierarchicalTokenizer(max_len=max_len)
    tok_data = Tokenizer.batch_encode(data)
    assert len(data)==len(tok_data), "Number of patients should be the same"
    for seq in tok_data:
        assert len(seq['codes']) == len(seq['visits'])  == len(seq['idx']) == len(seq['top_lvl_idx'])\
             == len(seq['ages']) == len(seq['los']) == len(seq['abs_pos']) == len(seq['values']), "All lists should have the same length"
        assert len(seq['codes']) <= max_len - len_background, "Sequence should be truncated"