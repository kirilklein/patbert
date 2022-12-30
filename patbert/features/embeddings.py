from torch import nn
import torch
import string
from patbert.common import medical

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = Embedding(config.vocab_size, 
                                            config.hidden_size, 
                                            padding_idx=config.pad_token_id)
        self.segment_embeddings = Embedding(config.seg_vocab_size, 
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

class Embedding(nn.Embedding):
    def forward(self, input):
        if isinstance(input, int):
            # handle integer input
            return self.weight[input]
        else:
            # handle tensor input
            return super(Embedding, self).forward(input)

class HierarchicalEmbedding(nn.Embedding):
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
        super().__init__(self.num_embeddings, self.embedding_dim)
        self.lab_test_vocab = self.get_lab_test_vocab()
        self.initialize_static_embeddings()

    def initialize_static_embeddings(self):
        self.top_lvl_embedding = Embedding(len(self.top_lvl_vocab), self.embedding_dim)

        self.birthdate_embedding = Embedding(self.num_birthyears, self.embedding_dim, _weight=torch.randn()) # number of birthmonths is smaller and so we can reuse the same embedding
        self.lab_test_embedding = Embedding(self.num_lab_tests+1, self.embedding_dim) # one for rare lab tests

        self.icd_atc_topic_embedding = Embedding(22, self.embedding_dim) # we add one topic for rare icd, atc contains 14 topics
        self.icd_atc_category_embedding = Embedding(len(string.ascii_uppercase)*10*10, self.embedding_dim) # this will cover all possible categories
        # TODO: split by subtopics
        self.icd_atc_subcategory_embedding = Embedding(len(string.ascii_uppercase)+10, self.embedding_dim) # all subcategories can be described by adding additional alphanumeric symbols
       
        self.top_lvl_embedding.weight.requires_grad = False
        self.birthdate_embedding .weight.requires_grad = False
        self.lab_test_embedding.weight.requires_grad = False
        self.icd_atc_category_embedding.weight.requires_grad = False
        self.icd_atc_topic_embedding.weight.requires_grad = False
        self.icd_atc_subcategory_embedding.weight.requires_grad = False


    def initialize_weights(self):
        initial_weights = torch.zeros(self.num_embeddings, self.embedding_dim)
        for token, index in self.vocab.items():
            initial_weights[int(index), :] = self.get_embedding(token)
        return initial_weights
    
    def get_embedding(self, token):
        """The embedding space will be divided in special tokens, birthyear, birthmonth, lab tests, icd, atc at the highest level"""
        if token in  ['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', '<MALE>', '<FEMALE>']:
            return self.top_lvl_embedding(self.top_lvl_vocab[token])
        elif token.startswith('BIRTHYEAR'):
            emb = self.top_lvl_embedding(self.top_lvl_vocab['<BIRTHYEAR>'])
            emb += self.birthdate_embedding(token.split('_')[1])/(2*self.kappa)
            return emb
        elif token.startswith('BIRTHMONTH'):
            emb = self.top_lvl_embedding(self.top_lvl_vocab['<BIRTHMONTH>'])
            emb += self.birthdate_embedding(token.split('_')[1])/(2*self.kappa)
            return emb
        elif token.startswith('D'):
            return self.icd_embedding(token)
        elif token.startswith('M'):
            return self.atc_embedding(token)
        elif token.startswith('L'):
            emb = self.top_lvl_embedding(self.top_lvl_vocab['L'])
            emb += self.lab_test_embedding(self.lab_test_vocab[token])/(2*self.kappa) # TODO: maybe we need to adjust for future lab tests
            return emb
        else:
            return self.top_lvl_embedding(self.top_lvl_vocab['<UNK>'])
    # TODO: add embedding for icd and atc

    def get_lab_test_vocab(self):
        ls = [x for x in self.vocab.keys() if x.startswith('L')]
        return {x: i for i, x in enumerate(ls)}

    def icd_embedding(self, token):
        emb = self.top_lvl_embedding(self.top_lvl_vocab['D'])
        emb += self.icd_atc_topic_embedding(medical.ICD_topic(token[1:]))/(2**self.kappa)
        if len(token) >= 4:
            emb += self.icd_atc_category_embedding(self.category2idx(token[1:]))/(3**self.kappa)
        else: return emb
        if len(token) >= 5:
            for i, alphanumeric in enumerate(token[4:]):
                emb += self.icd_atc_subcategory_embedding(self.alphanumeric2idx(alphanumeric))/((4+i)**self.kappa)
        return emb
    
    def atc_embedding(self, token):
        """The embedding is a sum of vectors which are decreasing in length with every level of the hierarchy.
        We start by modality (M), then topic e.g. A, B, C..., then category A01, A02,... , then every additional alphanumeric symbol is a step down the hierarchy."""
        emb = self.top_lvl_embedding(self.top_lvl_vocab['M'])
        emb += self.icd_atc_topic_embedding(medical.ATC_topic(token[1:]))/(2**self.kappa)
        if len(token) >= 4:
            emb += self.icd_atc_category_embedding(self.category2idx(token[1:]))/(3**self.kappa)
        else: return emb
        if len(token) >= 5:
            for i, alphanumeric in enumerate(token[4:]):
                emb += self.icd_atc_subcategory_embedding(self.alphanumeric2idx(alphanumeric))/((4+i)**self.kappa)
        return emb
    
    def category2idx(self, category):
        return string.ascii_uppercase.index(category[0])*100 + int(category[1:3])

    def alphanumeric2idx(self, alphanumeric):
        if alphanumeric.isnumeric():
            return int(alphanumeric)
        else:
            return string.ascii_uppercase.index(alphanumeric) + 10