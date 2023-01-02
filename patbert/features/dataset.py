from torch.utils.data.dataset import Dataset
import numpy as np
from patbert.features.utils import random_mask, seq_padding
import torch


class MLM_PLOS_Dataset(Dataset):
    def __init__(self, data, vocab, mask_prob=.15, pad_len=None):
        self.vocab = vocab
        lens = np.array([len(d['codes']) for d in data])
        max_len = int(np.max(lens)) + 5 # background sentence
        if pad_len is None:
            self.pad_len = max_len
        assert self.pad_len >= max_len, "pad_len must be larger than max_len"
        self.attributes = ['idx', 'codes', 'visits', 'los', 'abs_pos', 'age', 'values'] 
        self.data_all = [[d[attribute] for d in data] for attribute in self.attributes]
        self.mask_prob = mask_prob
    
    def __getitem__(self, index):
        """
        return: dictionary with codes for patient with index 
        """
        # 4th entry is now prolonged length of stay
        self.data_all[4][index] = (np.array(self.data_all[4][index])>7).astype(int)
        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.pad_len)
        mask[len(self.data_all[0][index]):] = 0
        # mask 
        masked_idxs, labels = random_mask(idxs, self.vocab, mask_prob=self.mask_prob) 
        # pad code sequence, segments and label
        for i in range(len(self.data_all)):
            self.data_all[i] = seq_padding(self.data_all[i], self.max_len, self.vocab)

        output_dic = {modality:torch.LongTensor(data) for \
            modality, data in zip(self.modalities, self.data_all)}
        return output_dic

    def __len__(self):
        return len(self.codes_all)

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

