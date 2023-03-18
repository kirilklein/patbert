

def zero_tuple(t, i):
    """Insert zero at index i in tuple t"""
    return t[:i] + (0,) + t[i+1:]

def map_concept_tuple(concept_tuple, tuple_vocab):
    """Giving a concept tuple, return the corresponding token."""
    n = 1
    while concept_tuple not in tuple_vocab and concept_tuple[0]!=0:
        new_concept_tuple = zero_tuple(concept_tuple, len(concept_tuple)-n)
        n+=1 
        concept_tuple = new_concept_tuple
    token = tuple_vocab[concept_tuple]
    return token