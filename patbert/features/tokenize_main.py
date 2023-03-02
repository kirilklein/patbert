from os.path import dirname, join, realpath
import os
import hydra
import torch
from patbert.features import tokenizer

base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')

@hydra.main(config_name='tokenize.yaml', config_path=config_path, version_base='1.3')
def tokenize(cfg):
    data = torch.load(cfg.data_path)
    Tokenizer = tokenizer.BaseTokenizer(pad_len=cfg.pad_len)
    tok_data = Tokenizer.batch_encode(data)
    torch.save(tok_data, join(os.getcwd(), 'tokenized_data.pt'))
    torch.save(Tokenizer.vocab, join(os.getcwd(), 'vocabulary.pt'))
    print(f"tokenized data and vocabulary saved in {os.getcwd()}")

if __name__ == "__main__":
    tokenize()
