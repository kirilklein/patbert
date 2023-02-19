import pandas as pd
import numpy as np
from hydra import utils as hydra_utils

class BaseProcessor():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def group_rare_values(self, df, col, category='OTHER'):
        """Rare values, below the threshold, are grouped into a single category,"""
        counts = df[col].value_counts(normalize=False)
        rare_values = counts[counts < self.cfg.patients_info.rare_threshold].index
        df.loc[df[col].isin(rare_values), col] = category
        return df

    
def value_process_identity(cfg, df, *args, **kwargs): # for compatibility with other classes
    return df

def value_process_binning(cfg, df):
    bins = hydra_utils.call(cfg.values_processing.binning_method, df=df)
    return bins

def fredman_diaconis_binning(values):
    max = values.max()
    min = values.min()
    IQR = np.percentile(values, 75) - np.percentile(values, 25)
    n = len(values)
    n_cr = n ** (1 / 3)
    bin_width = 2 * IQR / n_cr
    # n_bins = int(np.ceil(((max - min) / bin_width)))
    return np.arange(min, max+bin_width, bin_width)

def square_root_binning(values):
    max = values.max()
    min = values.min()
    sqn = np.sqrt(len(values))
    bin_width = (max-min) / sqn
    return np.arange(min, max+bin_width, bin_width)
