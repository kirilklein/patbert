import matplotlib.pyplot as plt
import numpy as np
from dirty_cat import SimilarityEncoder
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from collections import Counter


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