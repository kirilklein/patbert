
def get_parent(node_tuple):
    """Get parent node of a node defined by a tuple."""
    idx = next((i for i, x in enumerate(node_tuple) if x == 0), None)
    if idx is not None:
        return node_tuple[:idx - 1] + (0,) * (len(node_tuple) - idx + 1)
    else:
        return node_tuple[:-1] + (0,)

def get_leaf_nodes(tuple_dic):
    """
        Get the leaf nodes of a tree defined by a dictionary of tuples.
        Parameters: 
            tuple_dic: A dictionary of tuples, where the keys are the codes and the values are the tuples.
    """
    # Step 1: Create a set of parent nodes
    parent_nodes = set(get_parent(node_tuple) for node_tuple in tuple_dic.values())
    # Step 2: Identify leaf nodes
    leaf_nodes = {code: node_tuple for code, node_tuple in tuple_dic.items() if node_tuple not in parent_nodes}
    return leaf_nodes