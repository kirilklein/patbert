from torch import nn
import torch
import string
from patbert.common import medical

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
    def __init__(self, vocab, top_lvl_vocab, token2top_lvl, embedding_dim, kappa=2):
        # create the original embedding but with predefined weights
        self.kappa = kappa # controls the length of additional embeddings
        self.vocab = vocab
        self.top_lvl_vocab = top_lvl_vocab
        self.embedding_dim = embedding_dim
        self.token2top_lvl = token2top_lvl
        self.num_embeddings = len(vocab)
        
        self.num_birthyears = len([x for x in vocab.keys() if x.startswith('BIRTHYEAR')])
        self.num_lab_tests = len([x for x in vocab.keys() if x.startswith('L')])
        self.initialize_static_embeddings()
        

        super().__init__(self.num_embeddings, self.embedding_dim, _weight=self.initialize_weights())

    def initialize_static_embeddings(self):
        self.top_lvl_embedding = nn.Embedding(len(self.top_lvl_vocab), self.embedding_dim)

        self.birthyear_embedding = nn.Embedding(self.num_birthyears, self.embedding_dim)
        self.birthmonth_embedding = nn.Embedding(12, self.embedding_dim)

        self.lab_test_embedding = nn.Embedding(self.num_lab_tests+1, self.embedding_dim) # one for rare lab tests

        self.icd_topic_embedding = nn.Embedding(22, self.embedding_dim) # we add one topic for rare icd
        self.icd_atc_category_embedding = nn.Embedding(len(string.ascii_lowercase)*10*10, self.embedding_dim) # we add one topic for rare icd
        self.icd_subcategory_embedding = nn.Embedding(1000, self.embedding_dim) # 3 integers following category
       
        self.atc_topic_embedding = nn.Embedding(15, self.embedding_dim) # we add one topic for rare atc
        self.atc_subcategory_embedding = nn.Embedding(len(string.ascii_lowercase)+10, self.embedding_dim) # 3 integers following category

        self.top_lvl_embedding.weight.requires_grad = False
        self.birthyear_embedding .weight.requires_grad = False
        self.birthmonth_embedding.weight.requires_grad = False
        self.icd_topic_embedding.weight.requires_grad = False
        self.icd_atc_category_embedding.weight.requires_grad = False
        self.icd_subcategory_embedding.weight.requires_grad = False
        self.lab_test_embedding.weight.requires_grad = False
        self.atc_topic_embedding.weight.requires_grad = False
        self.atc_subcategory_embedding.weight.requires_grad = False


    def initialize_weights(self):
        initial_weights = torch.zeros(self.num_embeddings, self.embedding_dim)
        for token, index in enumerate(self.vocab.items()):
            initial_weights[index] = self.get_embedding(token)
        return initial_weights
    
    def get_embedding(self, token):
        if token in  ['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', '<MALE>', '<FEMALE>']:
            return self.top_lvl_embedding(self.top_lvl_vocab[token])
        elif token.startswith('BIRTHYEAR'):
            emb = self.top_lvl_embedding(self.top_lvl_vocab['BIRTHYEAR'])
            emb += self.birthyear_embedding(token.split('_')[1])/(2*self.kappa)
            return emb
        elif token.startswith('BIRTHMONTH'):
            emb = self.top_lvl_embedding(self.top_lvl_vocab['BIRTHMONTH'])
            emb += self.birthmonth_embedding(token.split('_')[1])/(2*self.kappa)
            return emb
        elif token.startswith('D'):
            return self.icd_embedding(token)
        elif token.startswith('M'):
            return self.atc_embedding(token)
        elif token.startswith('L'):
            emb = self.top_lvl_embedding(self.top_lvl_vocab['L'])
            emb += emb + self.lab_test_embedding(int(token[1:]))/(2*self.kappa) # TODO: maybe we need to adjust for future lab tests
            return emb
        else:
            return self.top_lvl_embedding(self.top_lvl_vocab['<UNK>'])
    # TODO: add embedding for icd and atc