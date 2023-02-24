import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from dirty_cat import SimilarityEncoder
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors


def similarity_encode(df, col):
    """Encode a column of strings using a 3gram similarity matrix"""
    X = df[col].sort_values().map(lambda x: x[1:]).unique() # strip prepend 
    enc = SimilarityEncoder()
    Y = enc.fit_transform(X.reshape(-1, 1))
    return X, Y, enc

def reduce_dim(Y):
    """Reduce dimensionality of similarity matrix using MDS"""
    mds =  MDS(dissimilarity="precomputed", n_init=10, random_state=42)
    Y2d = mds.fit_transform(1 - Y)
    return Y2d

def get_indices_to_visualize(enc, Y2d, n_points=100, n_neighbors=3):
    """Visualize n_neighbors nearest neighbors of n_points random points"""
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(Y2d)
    random_points = np.random.choice(
        len(enc.categories_[0]), n_points, replace=False)
    _, indices_ = nn.kneighbors(Y2d[random_points])
    indices = np.unique(indices_.squeeze())
    return indices

def visualize_encoded(df, col, n_points=100, n_neighbors=3):
    """Visualize a column of strings using a 3gram similarity matrix and dimensionality reduction"""
    X, Y, enc = similarity_encode(df, col)
    Y2d = reduce_dim(Y)
    indices = get_indices_to_visualize(enc, Y2d, n_points, n_neighbors)
    f, ax = plt.subplots()
    ax.scatter(x=Y2d[indices, 0], y=Y2d[indices, 1], s=.1)
    for x in indices: 
        ax.text(x=Y2d[x, 0], y=Y2d[x, 1], s=X[x], fontsize=8,)
    ax.set_title("multi-dimensional-scaling representation using a 3gram similarity matrix")

def zipfs_law_plot(df, ax, name, col='CONCEPT'):
    """PLot log-log of frequency of values in a column against log of number of values with that frequency"""
    counter = df[col].value_counts()
    counter_of_counts = Counter(counter.values)
    counts = np.array(list(counter_of_counts.keys()))
    freq_of_counts = np.array(list(counter_of_counts.values()))
    ax.scatter(np.log(counts), np.log(freq_of_counts))
    # set title location inside plot
    ax.set_title(f'{name}',  clip_on=False, y=.85)
    ax.set_xlabel(f'Log(freq.)')
    ax.set_ylabel(f'Log(number)')


def timing_function(function):
    """
    A decorator that prints the time a function takes to execute.
    """
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        print(f'{function.__qualname__!r}: {(t2 - t1)/60:.1f} mins')
        return result
    return wrapper

def sequence_train_test_split(sequence, config):
    train_dic, test_dic = {}, {}
    data_size = len(sequence['concept'])
    test_size = int(config.test_size * data_size)
    
    # split the data into train and test sets
    test_indices = np.random.choice(data_size, test_size, replace=False)
    train_indices = np.array(list(set(range(data_size)) - set(test_indices)))
    # create a dictionary with the train and test indices for each concept
    for key in sequence.keys():
        feature = np.array(sequence[key], dtype=object)
        train_dic[key] = feature[train_indices].tolist()
        test_dic[key] = feature[test_indices].tolist()
    return train_dic, test_dic