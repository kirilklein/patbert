from os.path import join, dirname, realpath

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra import utils as hydra_utils


class BaseProcessor():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    @staticmethod
    def convert_to_date(df, col):
        """Converts a column to datetime.date format"""
        df[col] = df[col].dt.date
    
    def write_concept_to_parquet(self, df):
        pq.write_table(pa.Table.from_pandas(df), join(os.getcwd(), f'concept.{self.concept}.parquet'))

class ValueProcessing:
    # value processing
    @staticmethod
    def value_process_identity(df, *args, **kwargs): # for compatibility with other methods
        return df
    @staticmethod
    def group_rare_values(df, cols, category='OTHER', rare_threshold=10):
        """Rare values, below the threshold, are grouped into a single category,"""
        for col in cols:
            counts = df[col].value_counts(normalize=False)
            rare_values = counts[counts < rare_threshold].index
            df.loc[df[col].isin(rare_values), col] = category
        return df
    @staticmethod
    def value_process_binning(df, cfg):
        """Continuous values of one concept are binned"""
        bins = hydra_utils.call(cfg.values_processing.binning_method, df=df)
        return bins

class BinningMethods:
    # binning methods
    @staticmethod
    def fredman_diaconis_binning(values):
        max = values.max()
        min = values.min()
        IQR = np.percentile(values, 75) - np.percentile(values, 25)
        n = len(values)
        n_cr = n ** (1 / 3)
        bin_width = 2 * IQR / n_cr
        # n_bins = int(np.ceil(((max - min) / bin_width)))
        return np.arange(min, max+bin_width, bin_width)
    @staticmethod
    def square_root_binning(values):
        max = values.max()
        min = values.min()
        sqn = np.sqrt(len(values))
        bin_width = (max-min) / sqn
        return np.arange(min, max+bin_width, bin_width)
