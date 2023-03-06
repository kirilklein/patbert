from patbert.losses.utils import get_leaf_node_probabilities
import torch
from torch.nn import NLLLoss
from torch import softmax

nllloss = NLLLoss()

def flat_softmax_cross_entropy(leaf_logits, y_true_enc, leaf_nodes):
    """Selects leaf probabilities for a given target tensor.
    Args:
        leaf_logits (torch.tensor): Logits (batchsize, num_leaf_nodes)
        y_true_enc (torch.tensor): Target vector (batchsize, seq_len, levels)
        leaf_nodes (torch.tensor): Leaf nodes (num_leaf_nodes, levels)
    Returns:
        Cross entropy loss"""
    leaf_probs = softmax(leaf_logits, dim=-1)
    selected_leaf_probs = get_leaf_node_probabilities(leaf_probs, y_true_enc, leaf_nodes)
    # print(selected_leaf_probs)
    log_probs = torch.log(selected_leaf_probs)
    log_probs = log_probs.flatten() # batchsize * seq_len
    loss = nllloss(log_probs.unsqueeze(-1), torch.zeros_like(log_probs, dtype=torch.int64))
    return loss