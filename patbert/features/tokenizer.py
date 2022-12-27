import torch
import typer
import string
import pickle as pkl
from os.path import join, split
from patbert.medical import icd


class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if isinstance(vocabulary, type(None)):
            self.vocabulary = {
                'CLS':0,
                'PAD':1,
                'SEP':2,
                'MASK':3,
                'UNK':4,
            }
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
        code_seqs = [seq[2] for seq in seqs] # icd codes
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
        tokenized_data_dic = {'pats':pat_ids, 'los':los_seqs, 'codes':output_code_seqs, 
                            'segments':output_visit_seqs}
        return tokenized_data_dic

    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)




class HierarchicalTokenizer():
    def __init__(self, vocabulary=None):
        super().__init__(vocabulary)
        
    def __call__(self, seq):
        return self.batch_encode(seq)

    def encode(self, seq):
        idx_seq  = []
        for code, mod in zip(seq['codes'], seq['modalities']):
            if mod == 'ICD10':
                topic = icd.ICD_topic(code)
                group = f"ICD10_{topic}"
            elif mod == 'ATC':
                group = f"ATC_{code[0]}"
            elif mod == 'LAB':
                group = code
            else:
                group = 'UNK'
            if group not in self.vocabulary:
                self.vocabulary[group] = len(self.vocabulary)
                idx_seq.append(self.vocabulary[group])
        return idx_seq

    def batch_encode(self, seqs, max_len=None):
        if isinstance(max_len, type(None)):
            max_len = max([len(seq) for seq in seqs])
        # let's do the padding later
        for seq in seqs:
            # truncation
            if len(seq['codes'])>max_len:
                codes = seq['codes'][-max_len:]
                modalities = seq['modalities'][-max_len:]
                seq['codes'] = codes
            # Tokenizing
            tokenized_code_seq = self.encode(code_seq)
            
        return tokenized_code_seq

    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)



def main(
    input_data_path: str = typer.Argument(..., 
        help="pickle list in the form [[pid1, los1, codes1, visit_nums1], ...]"),
    vocab_save_path: str = typer.Option(None, help="Path to save vocab, must end with .pt"),
    out_data_path: str = typer.Option(None, help="Path to save tokenized data, must end with .pt"),
    max_len: int = 
        typer.Option(None, help="maximum number of tokens to keep for each visit"),
    ):

    with open(input_data_path, 'rb') as f:
        data = pkl.load(f)

    Tokenizer = EHRTokenizer()
    tokenized_data_dic = Tokenizer.batch_encode(data, max_len=max_len)
    if isinstance(vocab_save_path, type(None)):
        data_dir = split(split(input_data_path)[0])[0]
        vocab_save_path = join(join(data_dir, 'tokenized', split(input_data_path)[1][:-4]+"_vocab.pt"))
    Tokenizer.save_vocab(vocab_save_path)
    if isinstance(out_data_path, type(None)):
        data_dir = split(split(input_data_path)[0])[0]
        out_data_path = join(join(data_dir, 'tokenized', split(input_data_path)[1][:-4]+"_tokenized.pt"))
    print(f"Save tokenized data to {out_data_path}")
    torch.save(tokenized_data_dic, out_data_path)
    
if __name__ == "__main__":
    typer.run(main)
