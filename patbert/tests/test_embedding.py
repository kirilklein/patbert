from operator import itemgetter

import torch

from patbert.common import medical
from patbert.features import embeddings


class Embedding_Tester():
    def __init__(self, embedding_dim=10) -> None:
        # TODO: we need to use real codes
        self.init_test_codes()
        vocab = {k:i for i,k in enumerate(self.test_codes)}
        self.vocab = vocab
        self.ids = self.get_ids_batch_from_codes()
        #_, vocab = common.load_data('synthetic')
        self.static_embedding = embeddings.StaticHierarchicalEmbedding(vocab, embedding_dim)
        self.embedding_dim = embedding_dim
        # If you want to change codes, append to this list
        # self.example_ids = torch.LongTensor(list(itemgetter(*self.test_codes)(vocab)))
        self.test_values = torch.randn_like(self.ids)
        self.mat = self.static_embedding(self.test_ids, self.test_values)
        print(self.mat.shape)
    def test_embedding_length(self):
        # TODO: we will need to provide values
        # TODO: here we can just select one batch for simplicity
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
    def test_value_multiplication(self):
        # embedding_mat

    def init_test_codes(self):
        medcodes = medical.MedicalCodes()
        icd = medcodes.get_icd()
        atc = medcodes.get_atc()
        self.test_codes  = [icd[1000], atc[2000], icd[300], atc[2000][:-2], 
            '<BIRTHYEAR>1950', '<CLS>']
    def get_ids_batch_from_codes(self):
        test_ids = [self.vocab[code] for code in self.test_codes]
        test_ids = torch.LongTensor(test_ids).reshape(2,3) # batch x len_seq
        return test_ids


def test(): 
    tester = Embedding_Tester()
    tester.test_embedding_length()
    #tester.test_embedding_shape()
    #tester.test_ids()
    