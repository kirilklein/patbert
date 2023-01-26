import numpy as np
from numpy.random import default_rng
from os.path import join, dirname, realpath
import torch
from patbert.common import medical, common
from patbert.features import tokenizer
import pickle as pkl
import hydra


def random_mask(idxs, vocab, mask_prob=0.15,
    special_tokens=['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', ], seed=0):
    """mask code with 15% probability, 80% of the time replace with [MASK], 
        10% of the time replace with random token, 10% of the time keep original"""
    rng = default_rng(seed)
    masked_idxs = idxs.copy()
    special_idxs = [vocab[token] for token in special_tokens]
    labels = len(idxs) * [-100] # -100 is ignored by loss function
    
    for i, idx in enumerate(idxs):
        if idx in special_idxs:
            continue
        prob = rng.uniform()
        if prob<mask_prob:
            prob = rng.uniform()  
            # 80% of the time replace with [MASK] 
            if prob < 0.8:
                masked_idxs[i] = vocab['<MASK>']
            # 10% change token to random token
            elif prob < 0.9:
                masked_idxs[i] = rng.choice(list(vocab.values())[len(special_idxs):]) # first tokens are special!
            # 10% keep original
            labels[i] = idx
    return masked_idxs, labels

def seq_padding(seq, max_len, vocab):
    """Pad a sequence to the given length."""
    return seq + (max_len-len(seq)) * [vocab['<PAD>']]

def get_int2int_dic_for_hembedings(vocab, num_levels=6):
    """Construct an integer to integer mapping from the vocab for Hierarchical Embeddings.
    """
    sks = medical.SKSVocabConstructor(vocab, num_levels=num_levels)
    vocabs = sks()
    list_of_dicts = [{} for _ in range(len(vocabs))]
    for k, v in vocab.items():
        for int2int, vocab in zip(list_of_dicts, vocabs):
            int2int[v] = vocab[k]
    return list_of_dicts


def remap_values(dic:dict, x:torch.Tensor)->torch.Tensor:
    """Helper function to remap values in a tensor using a dictionary."""
    remapping = torch.LongTensor(list(dic.keys())), torch.LongTensor(list(dic.values()))
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)

def get_data(cfg):
    """Loads processed data, performs tokenization and loads the tokenized data"""
    try:
        data = common.load_tokenized_data(cfg)
    except:
        # TODO we need to improve this by using hydra API
        data = create_tokenized_data(cfg)
    return data
    
def create_tokenized_data(cfg):
    """Creates tokenized data from processed data and saves it"""
    base_dir = dirname(dirname(dirname(realpath(__file__))))
    tokenized_dir = join(base_dir, 'data', 'tokenized')
    data_name = cfg.data.name
    proc_data = common.Data(cfg).load_processed_data()
    Tokenizer = hydra.utils.instantiate(cfg.data.tokenizer, cfg=cfg)
    tokenized_seq = Tokenizer.batch_encode(proc_data)
    torch.save(tokenized_seq, join(tokenized_dir, data_name + '.pt'))
    Tokenizer.save_vocab(join(tokenized_dir, data_name + '_vocab.pt'))
    if cfg.model.embedding.hierarchical:
        int2int  = get_int2int_dic_for_hembedings(
            Tokenizer.vocabulary, num_levels=cfg.model.embedding.num_levels)
        torch.save(int2int, join(tokenized_dir, data_name + '_hierarchy_mapping.pt'))
        return tokenized_seq, Tokenizer.vocabulary, int2int
    else:
        return tokenized_seq, Tokenizer.vocabulary


# Obsolete Functions
def combine_masks(mask1, mask2):
    """Combine two masks into one mask. 1 where either mask is 1, 0 otherwise"""
    # Initialize the output mask to all zeros
    mask3 = np.zeros_like(mask1)
    # Set the mask3 values to 1 where mask2 is 1 and 0 where mask1 is 1
    mask3[mask2] = 1
    mask3[mask1] = 0
    return mask3

def random_mask_arr(idxs, vocab, mask_prob=0.15,
    special_tokens=['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', ], seed=0):
    """is slower than random_mask with for loop even for sequences up to 1000 
        tokens, then only slightly faster"""
    rng = default_rng(seed)
    idxs = np.array(idxs)
    special_tokens += ['<SEX>0', '<SEX>1']
    special_tokens += [f'<BIRTHYEAR>{year}' for year in range(1900,2022)]
    special_tokens += [f'<BIRTHMONTH>{month}' for month in range(1,13)]
    special_idxs = np.array([vocab[token] for token in special_tokens])
    # Generate a mask indicating which tokens are special
    special_mask = np.isin(idxs, special_idxs)
    # Generate a random mask indicating which tokens should be masked
    modify_mask = rng.uniform(size=len(idxs)) < mask_prob
    # don't mask special tokens
    modify_mask = combine_masks(special_mask, modify_mask)
    # Mask the tokens that should be masked
    masked_idxs = idxs.copy()
    # Generate the labels for each token, -100 is ignored by loss function
    labels = np.ones_like(idxs)*(-100)
    labels[modify_mask] = idxs[modify_mask]
    # Randomly modify the masked tokens in the input
    prob = rng.uniform(size=len(idxs))
    mask_mask = modify_mask & (prob < 0.8)
    masked_idxs[mask_mask] = vocab['<MASK>']
    replace_mask = mask_mask & (prob >= 0.8) & (prob < 0.9)
    masked_idxs[replace_mask] = rng.choice(list(vocab.values()), 
        size=replace_mask.sum())
    return masked_idxs, labels