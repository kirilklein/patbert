from patbert.losses.utils import get_leaf_node_probabilities
import torch
from torch.nn import NLLLoss
from torch import softmax
from typing import List

nllloss = NLLLoss()

def flat_softmax_cross_entropy(leaf_logits, y_true_enc, leaf_nodes):
    """Selects leaf probabilities for a given target tensor.
    Args:
        leaf_logits (torch.tensor): Logits (batchsize, num_leaf_nodes)
        y_true_enc (torch.tensor): Target vector (batchsize, seq_len, levels)
        leaf_nodes (torch.tensor): Leaf nodes (num_leaf_nodes, levels)
    Returns:
        Cross entropy loss
    """
    leaf_probs = softmax(leaf_logits, dim=-1)
    selected_leaf_probs = get_leaf_node_probabilities(leaf_probs, y_true_enc, leaf_nodes)
    # print(selected_leaf_probs)
    log_probs = torch.log(selected_leaf_probs)
    log_probs = log_probs.flatten() # batchsize * seq_len
    loss = nllloss(log_probs.unsqueeze(-1), torch.zeros_like(log_probs, dtype=torch.int64))
    return loss







# For MultiLevelFlatSoftmaxCrossEntropyLoss
def get_target_child_mask(target_mask, y_flat):
    """Returns a mask that is true if the node is a child of the target"""
    target_mask_parent = target_mask[:,:,:-1].all(dim=2) & (~target_mask[:,:,-1])  # mask nodes that match target up to the last level 
    target_mask_parent = target_mask_parent & (y_flat[:,-1]==0) # on the last level the target is zero
    return target_mask_parent

def get_child_probabilities(target_child_mask):
    """Child probabilities are computed as 1/number of children"""
    return (1 / target_child_mask.sum(dim=0).float()).repeat(target_child_mask.shape[0], 1)

def get_target_mask(nodes_lvl, y_flat):
    """Returns a mask of shape num_nodes x num_targets x n_levels that is true if the node matches the target"""
    return nodes_lvl[:,None,:]==y_flat


def construct_target_probability_matrix(nodes, y_true_enc, level):
    """
    For a specific level, construct a matrix of target probabilities.
    If target is given on level: one hot encoding
    elif target is given on a higher level: probabilities split equally among all children
    else: zero
    Parameters: 
        nodes: list of tensors, where each tensor is a list of nodes for a specific level
        y_true_enc: target tensor of shape (batchsize, seq_len, levels) 
        level: level for which we want to construct the matrix (starts with 1)
    Returns:
        A: matrix of shape (batchsize x seq_len, num_nodes) where each row corresponds to a target and each column to a node on the given level"""
    nodes_lvl = nodes[level-1]
    num_nodes = nodes_lvl.shape[0]

    y_flat = torch.flatten(y_true_enc[:,:,:level], start_dim=0, end_dim=1) # flatten batch dim to simplify

    probability_matrix = torch.zeros((num_nodes, y_flat.shape[0])) # we populate this matrix num_nodes x num_targets

    target_mask = get_target_mask(nodes_lvl, y_flat) # mask nodes that match target, shape: num_nodes x num_targets x levels

    target_mask_exact = target_mask.all(dim=2) # mask nodes that match target exactly
    probability_matrix[target_mask_exact] = 1
    target_child_mask = get_target_child_mask(target_mask, y_flat)
    
    child_probabilities = get_child_probabilities(target_child_mask)

    probability_matrix[target_child_mask] = child_probabilities[target_child_mask]

    return probability_matrix.T

class FlatSoftmaxMultOP:
    def __init__(self, leaf_nodes, trainable_weights=0) -> None:
        self.leaf_nodes = leaf_nodes
        self.lvl_mappings = self.get_level_mappings()
        self.lvl_sel_mats = self.get_level_selection_mats()
        weights = self.initialize_geometric_weights()
        if trainable_weights:
            weights.requires_grad = True
        

    def __call__(self, predicted_probs:torch.tensor, y_true_enc:torch.tensor)->float:
        level_probs = []
        for i, mat in enumerate(self.lvl_sel_mats):
            print('mat', mat)
            print('pred', predicted_probs)
            prob_lvl = mat @ predicted_probs
            y_true_lvl = y_true_enc[:, :i+1]
            print('lvl_prob:', y_true_lvl)


    def initialize_geometric_weights(self):
        """We initialize weights as e**(-i)"""
        return torch.exp(-1*torch.arange(len(self.lvl_sel_mats)))

    def get_true_prob(level, y_true_enc):
        """Returns the probability for a given level and target"""
        pass

    def get_level_mappings(self):
        """Returns a dictionary, where the key is the level and the value is a tensor
        with the corresponding indices for this level.
        """
        lvl_mappings = []
        for i in range(len(self.leaf_nodes[0])):
            leaf_nodes_part = self.leaf_nodes[:,:i+1]
            _, unique_indices = torch.unique(leaf_nodes_part, dim=0, return_inverse=True) # we can use these to enumerate the target, in order to access the correct prob
            lvl_mappings.append([leaf_nodes_part, unique_indices])
        return lvl_mappings

    def get_level_selection_mats(self)->List[torch.tensor]:
        """For every level builds a matrix, such that when multiplied from the left
        with leaf probabilities, returns probabs for every node on this level. 
        Matrices are returned as list. 
        """
        mats = []
        nodes = []
        for i in range(len(self.leaf_nodes[0])):
            leaf_nodes_part = self.leaf_nodes[:,:i+1]
            # remove zero nodes (below leaf)
            unique_nodes, unique_indices = torch.unique(leaf_nodes_part, dim=0, return_inverse=True) # we can use these to enumerate the target, in order to access the correct prob
            # print(unique_nodes)
            # print('indices:', unique_indices)
            # Map each unique row to an integer based on its position in the sorted unique tensor
            mat = self.create_leaf_selection_matrix(unique_indices)
            # zero means we are below the leaf, so we set the corresponding row to zero
            zero_mask = leaf_nodes_part[:,-1]==0
            mat[:,zero_mask] = 0 
            all_zero_mask = mat.sum(dim=1)!=0
            mat = mat[all_zero_mask] # remove zero rows (corresponding to nodes from other levels)
            # mat = mat[mat.nonzero(dim=1)]
            unique_nodes = unique_nodes[all_zero_mask]
            mats.append(mat)
            nodes.append(unique_nodes)
        return mats, nodes
    # write a function that takes torch.tensor([1,1,1,2,2]) and returns a matrix that looks like this torch.tensor([[1,1,1,0,0],[0,0,0,1,1]])
    @staticmethod
    def create_leaf_selection_matrix(indices):
        """This function takes an array of integers and 
        returns a matrix where each row is a one-hot encoded version of the input.
        e,g, [1,1,1,2,2] -> [[1,1,1,0,0],[0,0,0,1,1]]]"""
        unique_values = torch.unique(indices)
        mask = (indices.unsqueeze(0) == unique_values.unsqueeze(1)).float()
        return mask