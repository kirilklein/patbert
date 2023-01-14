import torch
import typer
import pickle as pkl
from os.path import join, split, dirname, realpath
from collections import defaultdict

class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if isinstance(vocabulary, type(None)):
            self.special_tokens = ['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', 
                    '<MALE>', '<FEMALE>', '<BIRTHYEAR>', '<BIRTHMONTH>']
            self.vocabulary = {token:idx for idx, token in enumerate(self.special_tokens)}
        else:
            self.vocabulary = vocabulary

    def __call__(self, seq):
        return self.batch_encode(seq)

    def encode(self, seq):
        for code in seq:
            if code not in self.vocabulary:
                self.vocabulary[code] = len(self.vocabulary)
        return [self.vocabulary[code] for code in seq]

    def batch_encode(self, seqs, max_len=None):
        if isinstance(max_len, type(None)):
            max_len = max([len(seq) for seq in seqs])
        pat_ids = [seq[0] for seq in seqs]
        los_seqs = [seq[1] for seq in seqs]
        code_seqs = [seq[2] for seq in seqs] # icd code_ls
        visit_seqs = [seq[3] for seq in seqs]
        if isinstance(max_len, type(None)):
            max_len = max([len(seq) for seq in code_seqs])    
        output_code_seqs = []
        output_visit_seqs = []
        # let's do the padding later
        for code_seq, visit_seq in zip(code_seqs, visit_seqs):
            # truncation
            if len(code_seq)>max_len:
                code_seq = code_seq[:max_len]
                visit_seq = visit_seq[:max_len]
            # Tokenizing
            tokenized_code_seq = self.encode(code_seq)
            output_code_seqs.append(tokenized_code_seq)
            output_visit_seqs.append(visit_seq)
        tokenized_data_dic = {'pats':pat_ids, 'los':los_seqs, 'code_ls':output_code_seqs, 
                            'segments':output_visit_seqs}
        return tokenized_data_dic

    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)
    




class HierarchicalTokenizer():
    def __init__(self, vocabulary=None, max_len=None, len_background=5):
        """Background sentence is added later, so we need to know how many tokens it will have
        usually 5 (CLS, sex, birthyear, birthmonth, SEP)"""
        if isinstance(vocabulary, type(None)):
            self.special_tokens = ['<ZERO>','<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', 
                '<MALE>', '<FEMALE>', '<BIRTHYEAR>', '<BIRTHMONTH>']
            self.vocabulary = {token:idx for idx, token in enumerate(self.special_tokens)}
        else:
            self.vocabulary = vocabulary
        self.vocabs = [self.vocabulary]
        self.max_len = max_len
        self.len_background = len_background

    def __call__(self, seqs):
        return self.batch_encode(seqs)

    def batch_encode(self, seqs):
        seqs_new = []
        for seq in seqs:
            seqs_new.append(self.encode_seq(seq))
        return seqs_new
    
    def encode_seq(self, seq):
        self.enc_seq = defaultdict(list)
        self.seq = seq
        # it is easier to add the CLS token when concatenating with background sentence
        # since we cut off the sequence at the beginning, we will add the sep token later
        first_visit = 1
        abs_pos_0 = seq['abs_pos'][0]
        last_abs_pos = 0
        # skip pid, birthdate and sex
        for i in range(len(seq['codes'])):
        #for (code, age, los, visit, abs_pos, value) in zip(*list(seq.values())[3:]): 
            if seq['visits'][i]>first_visit:
                self.enc_seq['codes'].append('<SEP>')
                self.enc_seq['idx'].append(self.vocabulary['<SEP>'])
                self.enc_seq['visits'].append(first_visit)
                self.append_previous('ages')
                self.append_previous('abs_pos')
                self.enc_seq['los'].append(seq['abs_pos'][i-1]-abs_pos_0) # visit length in days         
                last_abs_pos = seq['abs_pos'][i]
                self.enc_seq['values'].append(1)
                first_visit = seq['visits'][i]
                if len(seq['abs_pos'])>(i+1):
                    abs_pos_0 = seq['abs_pos'][i+1] # abs pos of next visit
            # we will add the background sentence later, as it requires additional embeddings
            for key in ['visits', 'codes', 'ages', 'abs_pos', 'values']:
                self.append_token(key, i)
            # enc_seq['codes'].append(seq['codes'][i])
            self.enc_seq['los'].append(0)
            self.enc_seq['idx'].append(self.encode(seq['codes'][i]))
            # enc_seq['visits'].append(seq['visits'][i])
            # enc_seq['ages'].append(seq['ages'][i]), enc_seq['los'].append(seq['los'][i]), 
            # enc_seq['abs_pos'].append(seq['abs_pos'][i]), enc_seq['values'].append(seq['values'][i])
            if i==len(seq['codes'])-1:
                self.enc_seq['los'].append(seq['abs_pos'][i]-last_abs_pos)
        self.append_last_sep_token()
        self.truncate()
        self.insert_first_sep_token()
        return dict(self.enc_seq)

    def append_token(self, key, idx):
        self.enc_seq[key].append(self.seq[key][idx])

    def append_previous(self,key):
        self.enc_seq[key].append(self.enc_seq[key][-1])

    def insert_first_sep_token(self):
        """We insert zeros, sep and 1s, we might try a different value which will be mapped onto a zero vector"""
        self.enc_seq['codes'].insert(0, '<SEP>')
        self.enc_seq['idx'].insert(0, self.vocabulary['<SEP>'])
        self.enc_seq['values'].insert(0, 1)
        for key in ['visits', 'ages', 'abs_pos', 'los']:
            self.enc_seq[key].insert(0, 0)        

    def append_last_sep_token(self):
        self.enc_seq['codes'].append('<SEP>')
        self.enc_seq['idx'].append(self.vocabulary['<SEP>'])
        for key in ['visits', 'abs_pos', 'ages']:
            self.append_previous(key)
        self.enc_seq['values'].append(1)        

    def truncate(self):
        if not isinstance(self.max_len, type(None)):
            max_len = self.max_len - self.len_background # background sentence + CLS token
            if len(self.enc_seq['idx'])>max_len:
                if self.enc_seq['codes'][-max_len] == '<SEP>':
                    # we don't want to start a sequence with SEP
                    max_len = max_len - 1
                for key in ['codes', 'idx', 'values', 'visits', 'ages', 'abs_pos', 'los']:
                    self.truncate_key(key, max_len)
    
    def truncate_key(self, key, max_len):
        """Truncate one list inside seq"""
        self.seq[key] = self.seq[key][-max_len:]
        
    def encode(self, code):
        if code not in self.vocabulary:
            self.vocabulary[code] = len(self.vocabulary)
        return self.vocabulary[code]
    # add special tokens
    # check whether length is within limits
    # add background sentence
    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)




def main(
    input_data: str = typer.Argument(..., 
        help="data as generated by generate.py"),
    vocab_save_path: str = typer.Option(None, help="Path to save vocab, must end with .pt"),
    out_data_path: str = typer.Option(None, help="Path to save tokenized data, must end with .pt"),
    max_len: int = 
        typer.Option(None, help="maximum number of tokens to keep for each visit"),
    ):

    base_dir = dirname(dirname(dirname(realpath(__file__))))
    data_dir = join(base_dir, 'data')
    with open(join(data_dir, 'raw' , input_data + '.pkl'), 'rb')as f:
        data = pkl.load(f)

    Tokenizer = HierarchicalTokenizer(max_len=max_len)
    tokenized_data_dic = Tokenizer.batch_encode(data)
    if isinstance(vocab_save_path, type(None)):
        vocab_save_path = join(join(data_dir, 'vocabs', input_data + '.pt'))
    if isinstance(out_data_path, type(None)):
        out_data_path = join(join(data_dir, 'tokenized', input_data + '.pt'))
    Tokenizer.save_vocab(vocab_save_path)
    print(f"Save tokenized data to {out_data_path}")
    torch.save(tokenized_data_dic, out_data_path)
    
if __name__ == "__main__":
    typer.run(main)
