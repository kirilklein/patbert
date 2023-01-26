import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data.dataset import Dataset


class MLM_PLOS_Dataset(Dataset):
    def __init__(self, data, vocab, cfg):
        self.data = data
        self.vocab = vocab
        self.channels = cfg.data.channels
        self.plos_global = cfg.training.tasks.plos_global
        self.plos_threshold = cfg.training.tasks.plos_threshold
        self.init_pad_len(data, cfg.data.pad_len)
        self.mask_prob = cfg.data.mask_prob
        self.init_nonspecial_ids()
        self.pad_tokens = {'idx':self.vocab['<PAD>'],
                            'labels':-100,
                            'abs_pos':0,
                            'visits':0,
                            'ages':0,
                            'values':1,}
        
    def __getitem__(self, index):
        """
        return: dictionary with codes for patient with index 
        """ 
        pat_data = self.data[index]
        out_dic = {}
        if self.plos_global:
            out_dic['plos'] = int(any((np.array(pat_data['los'])>self.plos_threshold)))
        mask = self.get_mask(pat_data)
        out_dic['attention_mask'] = mask
        ids, labels = self.random_mask_ids(pat_data['idx']) 
        pat_data['values'] = self.mask_values(ids, pat_data['values']) # otherwise model could infer e.g. lab test
        # pad code sequence, segments and label
        #pat_data['codes'] = codes
        pat_data['idx'] = ids
        pat_data['labels'] = labels
        for channel in self.channels+['labels', 'idx']:
            out_dic[channel] = self.seq_padding(pat_data[channel], 
                self.pad_tokens[channel])    
            #print(channel, out_dic[channel])    
            out_dic[channel] = torch.LongTensor(out_dic[channel])    
            
        return out_dic

    def __len__(self):
        return len(self.data)
    
    def mask_values(self, ids, values):
        """Mask values the same way ids were masked"""
        mask_id = self.vocab['<MASK>']
        mask = np.array(ids)==mask_id
        values = np.array(values)
        values[mask] = 1
        return values.tolist()

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

    def init_nonspecial_ids(self):
        """We use by default < as special sign for special tokens"""
        self.nonspecial_ids = [v for k,v in self.vocab.items() if not k.startswith('<')]
        
    def seq_padding(self, seq, pad_token):
        """Pad a sequence to the given length."""
        return seq + (self.pad_len-len(seq)) * [pad_token]

    def random_mask_ids(self, ids, seed=0):
        """mask code with 15% probability, 80% of the time replace with [MASK], 
            10% of the time replace with random token, 10% of the time keep original"""
        rng = default_rng(seed)
        masked_ids = ids.copy()
        labels = len(ids) * [-100] 
        for i, id in enumerate(ids):
            if id not in self.nonspecial_ids:
                continue
            prob = rng.uniform()
            if prob<self.mask_prob:
                prob = rng.uniform()  
                # 80% of the time replace with [MASK] 
                if prob < 0.8:
                    masked_ids[i] = self.vocab['<MASK>']
                # 10% change token to random token
                elif prob < 0.9:      
                    masked_ids[i] = rng.choice(self.nonspecial_ids) 
                # 10% keep original
                labels[i] = id
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

