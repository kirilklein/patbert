from torch.utils.data.dataset import Dataset
import numpy as np
from patbert.features.utils import random_mask, seq_padding
import torch


class MLM_PLOS_Dataset(Dataset):
    def __init__(self, data, vocab, mask_prob=15):
        self.vocab = vocab
        self.codes_all = [d['codes'] for d in data]
        self.segments_all = [d['visits'] for d in data]
        self.los_all = [d['los'] for d in data]
        self.idx_all = [d['idx'] for d in data]
        self.abs_pos_all = [d['abs_pos'] for d in data]
        self.age_all = [d['age'] for d in data]
        self.values_all = [d['values'] for d in data]
        self.mask_prob = mask_prob
    def __getitem__(self, index):
        """
        return: code, position, visit, mask, label
        """
        idxs = self.idx_all[index]
        visits = self.visits_all[index]
        los = self.los_all[index]
        abs_pos = self.abs_pos_all[index]
        plos = (np.array(los)>7).any() #TODO: change to list for every visit
        # TODO: add age and values
        
        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(idxs):] = 0
        # mask 
        masked_idxs, labels = random_mask(idxs, self.vocab, mask_prob=self.mask_prob) 
        # pad code sequence, segments and label
        pad_idxs = seq_padding(masked_idxs, self.max_len, self.vocab)
        pad_visits = seq_padding(visits, self.max_len, self.vocab)
        pad_labels = seq_padding(labels, self.max_len, self.vocab)
        pad_abs_pos = seq_padding(abs_pos, self.max_len, self.vocab)
        output_dic = {
            'idxs':torch.LongTensor(pad_idxs),
            'visits':torch.LongTensor(pad_visits),
            'attention_mask':torch.LongTensor(mask),
            'labels':torch.LongTensor(pad_labels),
            'plos':torch.LongTensor(plos),
            'abs_pos':torch.LongTensor(pad_abs_pos),}
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

