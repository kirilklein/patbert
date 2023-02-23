import glob
import itertools
from datetime import datetime
import os
from os.path import join

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile
from patbert.data import utils


class BaseCreator():
    def __init__(self, config, test=False):
        self.config: dict = config
        self.test = test
        self.nrows = 100

    def __call__(self, concepts):
        return self.create(concepts)

    def create(self):
        raise NotImplementedError

    def datetime_conversion(self, df, datetime_cols):
        for col in datetime_cols:
            if col in df.keys():
                if not isinstance(df[col].dtype, datetime):
                    df[col] = pd.to_datetime(df[col])
        return df

    def read_file(self, cfg, file_path) -> pd.DataFrame:
        file_path = join(cfg.data_dir, file_path)
        file_type = file_path.split(".")[-1]
        datetime_cols = ['TIMESTAMP', 'BIRTHDATE', 'DEATHDATE']
        if file_type == 'csv':
            df = pd.read_csv(file_path, nrows= self.nrows if self.test else None)
        elif file_type == 'parquet':
            if not self.test:
                df = pd.read_parquet(file_path)
            else:
                pf = ParquetFile(file_path)
                batch = next(pf.iter_batches(batch_size = self.nrows)) 
                df = pa.Table.from_batches([batch]).to_pandas()
        else:
            raise ValueError(f'File type {file_type} not supported')
        return self.datetime_conversion(df, datetime_cols)
        

class ConceptCreator(BaseCreator):
    feature = 'concept'
    @utils.timing_function
    def create(self, concepts):
        # Get all concept files
        if not os.path.exists(self.config.data_dir):
            raise ValueError(f'Path {self.config.data_dir} does not exist')
        path = glob.glob('concept.*', root_dir=self.config.data_dir)
        # Filter out concepts files
        if self.config.get('concepts') is not None:
            path = [p for p in path if p.split('.')[1] in self.config.concepts]
        # Load concepts
        concepts = pd.concat([self.read_file(self.config, p) for p in path]).reset_index(drop=True)
        concepts = concepts.sort_values('TIMESTAMP')

        return concepts

class AgeCreator(BaseCreator):
    feature = 'age'
    @utils.timing_function
    def create(self, concepts):
        patients_info_file = glob.glob('patients_info.*', root_dir=self.config.data_dir)[0]
        patients_info = self.read_file(self.config, patients_info_file)
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate age
        # ages = concepts.apply(lambda x: ((x['TIMESTAMP'] - birthdates[x['PID']]).days // 365.25) if birthdates[x['PID']] is not None else -100, axis=1)
        # create ages series of length concepts filled with -100
        ages = pd.Series(np.full(len(concepts), -100))
        bd_mask = concepts['PID'].map(birthdates).notnull()
        ages.loc[bd_mask] = (concepts.loc[bd_mask, 'TIMESTAMP'] - concepts.loc[bd_mask, 'PID'].map(birthdates)).dt.days // 365.25
        # ages = (concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days // 365.25

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    feature = 'abspos'
    @utils.timing_function
    def create(self, concepts):
        abspos = self.config.abspos
        origin_point = datetime(abspos.year, abspos.month, abspos.day)
        # Calculate days since origin point
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.days

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = 'segment'
    @utils.timing_function
    def create(self, concepts):
        # Infer NaNs in ADMISSION_ID
        segments = concepts.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0]+1)

        concepts['SEGMENT'] = segments
        return concepts


class BackgroundCreator(BaseCreator):
    feature = 'background'
    @utils.timing_function
    def create(self, concepts):
        patients_info_file = glob.glob('patients_info.*', root_dir=self.config.data_dir)[0]
        patients_info = self.read_file(self.config, patients_info_file)

        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.features.background),
            'CONCEPT': itertools.chain.from_iterable([patients_info[col].tolist() for col in self.config.features.background])
        }

        for feature in self.config.features:
            if feature != self.feature:
                background[feature.upper()] = 0

        background = pd.DataFrame(background)

        return pd.concat([background, concepts])
        
