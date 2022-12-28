from torch import nn
import torch
import string

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, 
                                            config.hidden_size, 
                                            padding_idx=config.pad_token_id)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, 
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, code_ids, seg_ids):
        embeddings = self.word_embeddings(code_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        embeddings = embeddings + segment_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class HierarchicalEmbedding(nn.Module):
    def __init__(self, vocab, token_to_top_lvl, embedding_dim, num_lab_tests, embedding_matrix):
        super().__init__()
        self.ABC = string.ascii_uppercase
        self.initial_embedding = self.get_initial_embedding(vocab)
        self.first_level_embedding = nn.Embedding(2*self.ABC + num_lab_tests, embedding_dim)
        self.integer_embedding = nn.Embedding(10, embedding_dim)
        self.lab_test_embedding = nn.Embedding(num_lab_tests, embedding_dim)
        #self.letter_embedding.weight = nn.Parameter(embedding_matrix)
        self.first_level_embedding.weight.requires_grad = False
        self.integer_embedding.weight.requires_grad = False
        self.f1 = nn.Linear(embedding_dim, embedding_dim)
        self.gelu = nn.GELU()
        
    def forward(self, x, kappa=2):
        codes = x.codes
        mods = x.mods
        vals = x.vals
        icd_codes = codes[mods=='icd10']
        icd_inds = torch.nonzero(mods=='icd10')
        atc_codes = codes[mods=='atc']
        atc_inds = torch.nonzero(mods=='atc')
        lab_codes = codes[mods=='lab_test']
        lab_inds = torch.nonzero(mods=='lab_test')
        lab_vals = vals[mods=='lab_test']
        final_emb = torch.zeros(len(codes), self.embedding_dim)

        for ind, code in zip(icd_inds, icd_codes):
            final_emb[ind] = self.icd_embedding(code, kappa)
        for ind, code in zip(atc_inds, atc_codes):
            final_emb[ind] = self.icd_embedding(code, kappa)
        final_emb[lab_inds] = self.first_level_embedding(lab_codes, lab_vals) 
        return final_emb

    def icd_embedding(self, code, kappa=2):
        first_lvl_int = self.ABC.find(code[0]) + 5
        emb = self.first_level_embedding(first_lvl_int)
        for i, code in enumerate(code[1:]):
            emb += 1/(i+2)**kappa * self.integer_embedding(int(code))
        return emb
    
    def atc_embedding(self, code, kappa=2):
        first_lvl_int = self.ABC.find(code[0])*2 + 5
        emb = self.first_level_embedding(first_lvl_int)
        for i, code in enumerate(code[1:]):
            emb += 1/(i+2)**kappa * self.integer_embedding(int(code))
        return emb

    def lab_embedding(self, code, val):
        return self.first_level_embedding(code)*val 

class Hierarchical_Embedding(nn.Embedding):
    def __init__(self, vocab, token_to_top_lvl, embedding_dim):
        # create the original embedding but with predefined weights
        self.embedding_dim = embedding_dim
        self.token_to_top_lvl = token_to_top_lvl
        self.num_embeddings = len(vocab)
        super().__init__(self.num_embeddings, self.embedding_dim, _weight=self.initialize_weights())
        
    def initialize_weights(self):
        return torch.zeros(self.num_embeddings, self.embedding_dim)

