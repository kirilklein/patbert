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