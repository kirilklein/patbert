from operator import itemgetter

import torch

from patbert.common import medical
from patbert.features import embeddings


class Embedding_Tester():
    def __init__(self, embedding_dim=10) -> None:
        self.init_test_codes()
        vocab = {k:i for i,k in enumerate(self.test_codes)}
        #_, vocab = common.load_data('synthetic')
        self.static_embedding = embeddings.StaticHierarchicalEmbedding(vocab, embedding_dim)
        self.embedding_dim = embedding_dim
        # If you want to change codes, append to this list
        self.example_ids = torch.LongTensor(list(itemgetter(*self.test_codes)(vocab)))
        self.test_values = torch.randn(len(self.example_ids))
        self.mat = self.static_embedding(self.example_ids, self.test_values)
    def test_embedding_length(self):
        # TODO: we will need to provide values
        avg_lengths = torch.mean(torch.norm(self.mat, dim=2), dim=1)
        is_increasing = torch.all(torch.gt(avg_lengths[:-1], avg_lengths[1:]))
        assert is_increasing
    def test_embedding_shape(self):
        assert self.mat.shape[1] == len(self.test_codes), 'First dimension should be number of codes'
        assert self.mat.shape[2] == self.embedding_dim, 'Second dimension should be embedding dimension'
    def test_ids(self):
        ids_ls = self.static_embedding.get_ids_from_codes(self.test_codes)
        assert len(self.test_codes)==len(ids_ls[0]) == self.mat.shape[1], 'First dimension should be number of codes'
        assert ids_ls[0][0]==ids_ls[0][2], "ICD codes should have same id at level 0"
        for i in range(1, 5):
            assert ids_ls[i][6]==0, "Special tokens should have 0 at all levels except 0"
        first_zero = (ids_ls[:][1] != 0).sum()
        for i in range(first_zero-1):
            assert ids_ls[i][1]==ids_ls[i][3], "All indices until first zero have to be the same"
    def init_test_codes(self):
        sks = medical.SKSVocabConstructor()
        icd = sks.get_icd()
        atc = sks.get_atc()
        self.test_codes  = [icd[1000], atc[2000], icd[300], atc[2000][:-2], 
            '<BIRTHYEAR>1950', '<BIRTHMONTH>4', '<CLS>']


def test(): 
    tester = Embedding_Tester()
    tester.test_embedding_length()
    tester.test_embedding_shape()
    tester.test_ids()
    