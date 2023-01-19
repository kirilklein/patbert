from patbert.common import medical
from patbert.common import common

import os 
import torch
from os.path import dirname, realpath, join
base_dir = dirname(dirname(dirname(realpath(__file__))))

main_vocab = torch.load(join(base_dir, 'data', 'vocabs', 'synthetic.pt'))
sks = medical.SKSVocabConstructor(main_vocab)


def test_topics():
    vocab = sks.construct_vocab_dic(1)
    # test some random topics
    for topic in ['DA', 'DK', 'DU', 'DV', 'MA', 'ML', 'MG']:
        v = {k:v for k,v in vocab.items() if k.startswith(topic)}
        assert common.check_same_elements(list(v.values())), 'All values should be the same'

def test_vocab():
    for level in range(2,5):
        vocab = sks.construct_vocab_dic(level)
        if level==2:
            for category in ['DG30', 'DUH', 'DUA', 'MA03', 'MH01']:
                v = {k:v for k,v in vocab.items() if k.startswith(category)}
                assert common.check_same_elements(list(v.values())), 'All values should be the same'
        # get all codes that start with DU followed by a digit
        vDU = {k:v for k,v in vocab.items() if k.startswith('DU') and k[2].isdigit()}
        vDV = {k:v for k,v in vocab.items() if k.startswith('DV') and k[2].isdigit()}
        if level==2:
            assert common.check_same_elements(list(vDU.values())), 'All values should be the same'
            assert common.check_same_elements(list(vDV.values())), 'All values should be the same'
        else:
            assert all([value==0 for value in vDU.values()]), 'Values should be 0'
            assert all([value==0 for value in vDV.values()]), 'Values should be 0'
        if level==3:
            v = {k:v for k,v in vocab.items() if k.startswith('DVA')}
            assert common.check_unique(list(v.values())), 'All values should be unique'
            # check uniqueness of values for some topics
            # values = []
            # for topic in ['DG', 'DU', 'DV', 'MA', 'ML', 'MG']:
                # values.append([v for k,v in vocab.items() if k.startswith(topic)][0])
                # assert common.check_unique(values), 'All values should be unique'    
        if level==5:
            v = common.key_length(vocab, 5)
            for end_int in ['01', '02', '03']:
                v_int = common.inspect_dic(v, 'M',end_int) # medical
                assert common.check_same_elements(list(v_int.values())), 'at lower levels same pattern should have same index'


                
    