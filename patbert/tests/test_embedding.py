from patbert.features import embeddings
from patbert.common import medical
import torch


class Embedding_Tester():
    def __init__(self, embedding_dim=10) -> None:
        self.static_embedding = embeddings.StaticHierarchicalEmbedding(embedding_dim)
        sks = medical.SKSVocabConstructor()
        icd = sks.get_icd()
        atc = sks.get_atc()
        # If you want to change codes, append to this list
        self.test_codes  = [icd[1000], atc[2000], icd[300], atc[2000][:-2], 
            '<BIRTHYEAR>1950', '<BIRTHMONTH>4', '<CLS>']
        self.mat = self.static_embedding(self.test_codes, [])
    def test_embedding_length(self):
        # TODO: we will need to provide values
        avg_lengths = torch.mean(torch.norm(self.mat, dim=2), dim=1)
        is_increasing = torch.all(torch.gt(avg_lengths[:-1], avg_lengths[1:]))
        assert is_increasing
    def test_ids(self):
        ids_ls = self.static_embedding.get_ids_from_codes(self.test_codes)
        print(ids_ls)
        assert len(self.test_codes)==len(ids_ls[0]) == self.mat.shape[1], 'First dimension should be number of codes'
        assert ids_ls[0][0]==ids_ls[0][2], "ICD codes should have same id at level 0"
        for i in range(1, 5):
            assert ids_ls[i][6]==0, "Special tokens should have 0 at all levels except 0"
        first_zero = (ids_ls[:][1] != 0).sum()
        for i in range(first_zero-1):
            assert ids_ls[i][1]==ids_ls[i][3], "All indices until first zero have to be the same"
tester = Embedding_Tester()
tester.test_embedding_length()
tester.test_ids()