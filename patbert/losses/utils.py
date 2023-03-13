from torch import tensor
import torch


# should be part of a class, leaf nodes stay the same
def get_leaf_node_probabilities(leaf_probs:torch.tensor, y_true_enc: torch.tensor, leaf_nodes: torch.tensor):
    """Selects leaf probabilities for a given target tensor.
    Args:
        leaf_probs (torch.tensor): Probabilities (batchsize, num_leaf_nodes)
        y_true_enc (torch.tensor): Target vector (batchsize, seq_len, levels)
        leaf_nodes (torch.tensor): Leaf nodes (num_leaf_nodes, levels)
    Returns:
        torch.tensor: Selected leaf probabilities (batchsize, seq_len)"""
    # we want to match all the leaf nodes with a target, e.g. target: 1,2,0 should select 1,2,1 and 1,2,2
    zeros_mask = y_true_enc == 0
    leaf_mask = ( leaf_nodes == y_true_enc[:, :, None, :] )| zeros_mask[:, :, None,:] # select all leaf nodes that match the target
    leaf_mask = leaf_mask.all(dim=-1).to(torch.int16)

    leaf_probs = leaf_probs[:,None,:].expand(leaf_mask.shape) # batch, seq_len, num_leafes

    selected_leaf_probs = leaf_probs * leaf_mask
    selected_leaf_probs = selected_leaf_probs.sum(dim=-1)
    return selected_leaf_probs