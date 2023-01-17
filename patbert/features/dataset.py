import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data.dataset import Dataset

from patbert.features import utils


class MLM_PLOS_Dataset(Dataset):
    def __init__(self, data, vocab, channels=['visits', 'abs_pos', 'ages'], 
            mask_prob=.15, pad_len=None, plos=False):
        self.data = data
        self.vocab = vocab
        self.channels = channels
        self.plos = plos
        self.init_pad_len(data, pad_len)
        self.mask_prob = mask_prob
        self.init_nonspecial_codes()
        
    def __getitem__(self, index):
        """
        return: dictionary with codes for patient with index 
        """ 
        pat_data = self.data[index]
        out_dic = {}
        if self.plos:
            out_dic['plos'] = int(any((np.array(pat_data['los'])>7)))
        mask = self.get_mask()
        out_dic['attention_mask'] = mask
        codes, ids, labels = self.random_mask_codes_ids(pat_data['codes'], pat_data['idx']) 
        # pad code sequence, segments and label
        #pat_data['codes'] = codes
        pat_data['idx'] = ids
        pat_data['labels'] = labels
        pad_tokens = [0, 0, 0, -100, self.vocab['<PAD>'], '<PAD>'] # other channels need different padding
        for k, pad_token in zip(['visits','abs_pos', 'ages', 'labels', 'idx', 'codes'], pad_tokens):
            out_dic[k] = utils.seq_padding(pat_data[k], pad_token)        
        for key in ['visits','abs_pos', 'ages', 'labels', 'idx']:
            out_dic[key] = torch.LongTensor(out_dic[key])    
            
        return out_dic

    def __len__(self):
        return len(self.data)
    

    def get_mask(self, pat_data):
        mask = np.ones(self.pad_len)
        mask[len(pat_data['codes']):] = 0
        return mask

    def init_pad_len(self, data, pad_len):
        if isinstance(pad_len, type(None)):
            lens = np.array([len(d['codes']) for d in data])
            self.pad_len = int(np.max(lens)) 
        else:
            self.pad_len = pad_len

    def init_nonspecial_codes(self):
        special_tokens = [tok for tok in self.vocab.keys() if tok.startswith('<')]
        special_idxs = [self.vocab[token] for token in special_tokens]
        self.nonspecial_codes = [k for k, v in self.vocab.items() if v not in special_idxs]
        
    def seq_padding(self, seq, pad_token):
        """Pad a sequence to the given length."""
        return seq + (self.pad_len-len(seq)) * [pad_token]

    def random_mask_codes_ids(self, ids, seed=0, ):
        """mask code with 15% probability, 80% of the time replace with [MASK], 
            10% of the time replace with random token, 10% of the time keep original"""
        rng = default_rng(seed)
        # masked_codes = codes.copy()
        masked_ids = ids.copy()
        # TODO: this needs to be improved
        labels = len(ids) * [-100] 
        
        for i, code in enumerate(ids):
            if code not in self.nonspecial_codes:
                continue
            prob = rng.uniform()
            if prob<self.mask_prob:
                prob = rng.uniform()  
                # 80% of the time replace with [MASK] 
                if prob < 0.8:
                    # masked_codes[i] = '<MASK>'
                    masked_ids[i] = self.vocab['<MASK>']
                # 10% change token to random token
                elif prob < 0.9:      
                    random_code = rng.choice(self.nonspecial_codes)          
                    # masked_codes[i] = random_code# first tokens are special!
                    masked_ids[i] = self.vocab[random_code]
                # 10% keep original
                labels[i] = self.vocab[code]
        return masked_ids, labels

class PatientDatum():
    def __init__(self, data, vocab, pat_id):
        self.vocab = vocab
        self.codes = data['codes'][pat_id]
        self.segments = data['segments'][pat_id]

    def __getitem__(self, index):
        """
        return: code, position, segmentation, mask, label
        """
        output_dic = {
            'codes':torch.LongTensor(self.codes),
            'segments':torch.LongTensor(self.segments),}
        return output_dic

    def __len__(self):
        return 1

