import torch
import typer
import string
import pickle as pkl
from os.path import join, split
from patbert.common.medical import ICD_topic


class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if isinstance(vocabulary, type(None)):
            special_tokens = ['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', '<MALE>', '<FEMALE>']
            birthyear_tokens = [f'BIRTHYEAR_{year}' for year in range(1900, 2022)]
            birthmonth_tokens = [f'BIRTHMONTH_{month}' for month in range(1, 13)]
            self.special_tokens = special_tokens + birthyear_tokens + birthmonth_tokens
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
    




class HierarchicalTokenizer(EHRTokenizer):
    def __init__(self, vocabulary=None, max_len=None, len_background=5):
        """Background sentence is added later, so we need to know how many tokens it will have
        usually 5 (CLS, sex, birthyear, birthmonth, SEP)"""
        super().__init__(vocabulary)
        self.top_lvl_vocab = self.vocabulary.copy()
        self.max_len = max_len
        self.len_background = len_background
        self.token2top_lvl = {}
    def __call__(self, seqs):
        return self.batch_encode(seqs)

    def batch_encode(self, seqs):
        seqs_new = []
        for seq in seqs:
            idx_ls, top_level_idx_ls = [], []
            code_ls, age_ls, los_ls, visit_ls, abs_pos_ls, mod_ls, value_ls = [[] for _ in range(7)]
            # it is easier to add the CLS token when concatenating with background sentence
            # since we cut off the sequence at the beginning, we will add the sep token later
           
            first_visit = 1
            
            for (code, age, los, visit, abs_pos, value) in zip(*list(seq.values())[3:]):
                if visit>first_visit:
                    idx_ls.append(self.vocabulary['<SEP>']), top_level_idx_ls.append(self.top_lvl_vocab['<SEP>'])
                    code_ls.append('<SEP>'), mod_ls.append('<SEP>')
                    visit_ls.append(first_visit)
                    age_ls.append(age_ls[-1]), los_ls.append(los_ls[-1]), abs_pos_ls.append(abs_pos_ls[-1]) 
                    value_ls.append(1)
                    first_visit = visit        
                # we will add the background sentence later, as it requires additional embeddings
                code_ls.append(code)
                idx_ls.append(self.encode(code)), top_level_idx_ls.append(self.encode_top_lvl(code))
                visit_ls.append(visit)
                age_ls.append(age), los_ls.append(los), abs_pos_ls.append(abs_pos), value_ls.append(value)
            
            seq['codes'] = code_ls
            seq['idx'] = idx_ls
            seq['top_lvl_idx'] = top_level_idx_ls
            seq['ages'] = age_ls
            seq['los'] = los_ls 
            seq['visits'] = visit_ls 
            seq['abs_pos'] = abs_pos_ls
            seq['values'] = value_ls
            seq = self.append_last_sep_token(seq)
            seq = self.truncate(seq)
            seq = self.insert_first_sep_token(seq)
            seqs_new.append(seq)
        return seqs_new
    
    def insert_first_sep_token(self, seq):
        """We insert zeros, sep and 1s, we might try a different value which will be mapped onto a zero vector"""
        seq['codes'].insert(0, '<SEP>')
        seq['idx'].insert(0, self.vocabulary['<SEP>'])
        seq['top_lvl_idx'].insert(0, self.vocabulary['<SEP>'])
        seq['visits'].insert(0, 0)
        seq['abs_pos'].insert(0, 0)
        seq['ages'].insert(0, 0)
        seq['values'].insert(0, 1)
        seq['los'].insert(0, 0)
        return seq

    def append_last_sep_token(self, seq):
        seq['codes'].append('<SEP>')
        seq['idx'].append(self.vocabulary['<SEP>'])
        seq['top_lvl_idx'].append(self.vocabulary['<SEP>'])
        seq['visits'].append(seq['visits'][-1])
        seq['abs_pos'].append(seq['abs_pos'][-1])
        seq['ages'].append(seq['ages'][-1])
        seq['values'].append(1)
        seq['los'].append(seq['los'][-1])
        return seq

    def truncate(self, seq):
        if not isinstance(self.max_len, type(None)):
            max_len = self.max_len - self.len_background # background sentence + CLS token
            if len(seq['idx'])>max_len:
                if seq['codes'][-max_len] == '<SEP>':
                    # we don't want to start a sequence with SEP
                    max_len = max_len - 1
                seq['codes'] = seq['codes'][-max_len:]
                seq['idx'] = seq['idx'][-max_len:]
                seq['top_lvl_idx'] = seq['top_lvl_idx'][-max_len:]
                seq['ages'] = seq['ages'][-max_len:]
                seq['los'] = seq['los'][-max_len:]
                seq['visits'] = seq['visits'][-max_len:]
                seq['abs_pos'] = seq['abs_pos'][-max_len:]
                seq['values'] = seq['values'][-max_len:]
        return seq

    def encode(self, code):
        if code not in self.vocabulary:
            self.vocabulary[code] = len(self.vocabulary)
        return self.vocabulary[code]
        
    def encode_top_lvl(self, code):
        """Encode top level (first level in hierarchy) token for a given code and modality"""
        if code[0] in "MDL": # M: ATC, D: ICD, L: LAB
            group = code[0]
        elif code.startswith('BIRTHYEAR'):
            group = 'BIRTHYEAR'
        elif code.startswith('BIRTHMONTH'):
            group = 'BIRTHMONTH'
        else:
            if code not in self.special_tokens:
                Warning(f"Modality of code {code} not recognized, set to unknown (<UNK>)")
                group = '<UNK>'
            else:
                group = code
        if group not in self.top_lvl_vocab:
            self.top_lvl_vocab[group] = len(self.top_lvl_vocab)
        if code not in self.token2top_lvl:
            self.token2top_lvl[code] = group
        return self.top_lvl_vocab[group]

    # add special tokens
    # check whether length is within limits
    # add background sentence
    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)
        torch.save(self.top_lvl_vocab, dest.replace('.pt', '_top_lvl.pt'))
        torch.save(self.token2top_lvl, dest.replace('.pt', '_token2top_lvl.pt'))
        torch.save(self.special_tokens, dest.replace('.pt', '_special_tokens.pt'))



def main(
    input_data_path: str = typer.Argument(..., 
        help="data as generated by generate.py, must end with .pkl"),
    vocab_save_path: str = typer.Option(None, help="Path to save vocab, must end with .pt"),
    out_data_path: str = typer.Option(None, help="Path to save tokenized data, must end with .pt"),
    max_len: int = 
        typer.Option(None, help="maximum number of tokens to keep for each visit"),
    ):

    with open(input_data_path, 'rb') as f:
        data = pkl.load(f)

    Tokenizer = HierarchicalTokenizer(max_len=max_len)
    tokenized_data_dic = Tokenizer.batch_encode(data)
    if isinstance(vocab_save_path, type(None)):
        data_dir = split(split(input_data_path)[0])[0]
        vocab_save_path = join(join(data_dir, 'vocabs', split(input_data_path)[1].replace('.pkl', '.pt')))
    Tokenizer.save_vocab(vocab_save_path)
    if isinstance(out_data_path, type(None)):
        data_dir = split(split(input_data_path)[0])[0]
        out_data_path = join(join(data_dir, 'tokenized', split(input_data_path)[1].replace('.pkl', '.pt')))
    print(f"Save tokenized data to {out_data_path}")
    torch.save(tokenized_data_dic, out_data_path)
    
if __name__ == "__main__":
    typer.run(main)
