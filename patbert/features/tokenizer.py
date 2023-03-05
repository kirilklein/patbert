import pickle as pkl
from collections import defaultdict
from os.path import dirname, join, realpath

import torch
import typer
from tqdm import tqdm

from patbert.features import utils


class BaseTokenizer():
    def __init__(self, pad_len, vocabulary=None):
        """Add description"""
        if isinstance(vocabulary, type(None)):
            self.special_tokens = ['[ZERO', '[CLS]',
                                   '[PAD]', '[SEP]', '[MASK]', '[UNK]',]
            self.vocabulary = {token: idx for idx,
                               token in enumerate(self.special_tokens)}
        else:
            self.vocabulary = vocabulary
        self.max_len = pad_len

    def __call__(self, seqs):
        return self.batch_encode(seqs)

    def batch_encode(self, seqs):
        seqs['idx'] = []
        for i in tqdm(range(len(seqs['concept']))):
            seqs['idx'].append(self.encode_seq(seqs['concept'][i]))
        del seqs['concept']
        return seqs

    def encode_seq(self, seq):
        ls = []
        for code in seq:
            ls.append(self.encode(code))
        return ls

    def encode(self, code):
        """Encode a code, if it is not in the vocabulary, add it."""
        if code not in self.vocabulary:
            self.vocabulary[code] = len(self.vocabulary)
        return self.vocabulary[code]


class EHRTokenizer(BaseTokenizer):
    """EHR tokenizer with background sentent and different channels"""

    def __init__(self, pad_len, cfg, vocabulary=None, len_background=5):
        super(self, EHRTokenizer).__init__(pad_len, cfg)
        """Add description"""
        if isinstance(vocabulary, type(None)):
            self.special_tokens = [
                '[ZERO]',
                '[CLS]',
                '[PAD]',
                '[SEP]',
                '[MASK]',
                '[UNK]',
            ]
            self.vocabulary = {token: idx for idx,
                               token in enumerate(self.special_tokens)}
            for i in range(1900, 2022):
                self.vocabulary[f'[BIRTHYEAR]{i}'] = len(self.vocabulary)
            for i in range(1, 13):
                self.vocabulary[f'[BIRTHMONTH]{i}'] = len(self.vocabulary)
        else:
            self.vocabulary = vocabulary
        self.len_background = len_background  # usually cls, sex, birthyear, birthmonth

    def encode_seq(self, seq):
        self.enc_seq = defaultdict(list)  # we need a dictionary of lists
        self.seq = seq
        first_visit = 1
        abs_pos_0 = seq['abs_pos'][0]
        last_abs_pos = 0

        # skip pid, birthdate and sex
        for i in range(len(seq['concept'])):
            if seq['segment'][i] > first_visit:
                self.append_code_idx('[SEP]')
                self.enc_seq['segment'].append(first_visit)
                self.append_previous_token('ages')
                self.append_previous_token('abs_pos')
                self.enc_seq['los'].append(
                    seq['abs_pos'][i - 1] - abs_pos_0)  # visit length in days
                last_abs_pos = seq['abs_pos'][i]
                self.enc_seq['values'].append(1)
                first_visit = seq['segment'][i]
                if len(seq['abs_pos']) > (i + 1):
                    abs_pos_0 = seq['abs_pos'][i + 1]  # abs pos of next visit
            for key in ['segment', 'concept', 'ages', 'abs_pos', 'values']:
                self.append_token_from_original(key, i)
            self.enc_seq['los'].append(0)
            self.enc_seq['idx'].append(self.encode(seq['concept'][i]))
            if i == (len(seq['concept']) - 1):
                self.enc_seq['los'].append(seq['abs_pos'][i] - last_abs_pos)
        self.append_last_sep_token()
        self.truncate()
        self.insert_first_sep_token()
        self.construct_background_sentence()
        return dict(self.enc_seq)

    def append_code_idx(self, code):
        """We append code to codes and the corresponding index to idx."""
        self.enc_seq['concept'].append(code)
        self.enc_seq['idx'].append(self.encode(code))

    def construct_background_sentence(self):
        birthdate = self.seq['birthdate']
        sex = self.seq['sex']
        background_codes = [f'[BIRTHMONTH>{birthdate.month}',
                            f'[BIRTHYEAR>{birthdate.year}', f'[SEX>{sex}', '[CLS]']  # order important
        for code in background_codes:
            self.insert_code_idx(code)

        for key in ['segment', 'ages',
                    'abs_pos', 'los']:  # fill other lists with zeros
            self.insert_values(len(background_codes), key, 0)
        self.insert_values(len(background_codes), 'values',
                           1)  # fill values with 1

    def insert_code_idx(self, code):
        """We insert code to codes and the corresponding index to idx."""
        self.enc_seq['concept'].insert(0, code)
        self.enc_seq['idx'].insert(0, self.encode(code))

    def insert_values(self, n, key, value=0):
        """n: Number of values at start of sequence to insert"""
        ins_ls = [value] * n
        self.enc_seq[key] = ins_ls + self.enc_seq[key]

    def append_token_from_original(self, key, idx):
        """Get the token at the given index from the original seq
            and append it to the enc_seq to the <key> list."""
        self.enc_seq[key].append(self.seq[key][idx])

    def append_previous_token(self, key):
        """Append the previous token to the enc_seq dictionary."""
        self.enc_seq[key].append(self.enc_seq[key][-1])

    def insert_first_sep_token(self):
        """We insert zeros, sep and 1s, we might try a different value which will be mapped onto a zero vector"""
        self.insert_code_idx('[SEP]')
        self.enc_seq['values'].insert(0, 1)
        for key in ['segment', 'ages', 'abs_pos', 'los']:
            self.enc_seq[key].insert(0, 0)

    def append_last_sep_token(self):
        """Append the last sep token to the enc_seq dictionary.
        Except for codes and indices with append the previous token, for values append 1"""
        self.append_code_idx('[SEP]')
        for key in ['segment', 'abs_pos', 'ages']:
            self.append_previous_token(key)
        self.enc_seq['values'].append(1)

    def truncate(self):
        """Truncate the sequence to max_len-len_background,
            since we add a background sentence in the next step"""
        if not isinstance(self.max_len, type(None)):
            max_len = self.max_len - self.len_background  # background sentence + CLS token
            if len(self.enc_seq['idx']) > max_len:
                if self.enc_seq['concept'][-max_len] == '[SEP]':
                    # we don't want to start a sequence with SEP
                    max_len = max_len - 1
                for key in ['concept', 'idx', 'values',
                            'segment', 'ages', 'abs_pos', 'los']:
                    self.truncate_ls(key, max_len)

    def truncate_ls(self, key, max_len):
        """Truncate one list inside seq"""
        self.enc_seq[key] = self.enc_seq[key][-max_len:]

    # add special tokens
    # check whether length is within limits
    # add background sentence

    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)


