import glob
import itertools
from datetime import datetime
import os
from os.path import join

import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile


class BaseCreator():
    def __init__(self, config, test=False):
        self.config: dict = config
        self.test = test
        self.nrows = 10000 

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
            df = self.datetime_conversion(df, datetime_cols)
            return df
        elif file_type == 'parquet':
            if not self.test:
                df = pd.read_parquet(file_path)
                self.datetime_conversion(df, datetime_cols)
                print(df.dtypes)
                assert False
                return  df
            else:
                pf = ParquetFile(file_path)
                batch = next(pf.iter_batches(batch_size = int(1e5))) 
                df = pa.Table.from_batches([batch]).to_pandas()
                df = self.datetime_conversion(df, datetime_cols)
                return  df
        else:
            raise ValueError(f'File type {file_type} not supported')

class ConceptCreator(BaseCreator):
    feature = 'concept'
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
        # check if TIMESTAMP is of datetime type, if not convert it
        if not isinstance(concepts.TIMESTAMP[0], datetime):
            concepts['TIMESTAMP'] = pd.to_datetime(concepts['TIMESTAMP'].str.slice(stop=10))
        concepts = concepts.sort_values('TIMESTAMP')

        return concepts

class AgeCreator(BaseCreator):
    feature = 'age'
    def create(self, concepts):
        patients_info_file = glob.glob('patients_info.*', root_dir=self.config.data_dir)[0]
        patients_info = self.read_file(self.config, patients_info_file)
        print(patients_info.head())
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate age
        print(concepts.head())
        ages = concepts.apply(lambda x: ((x['TIMESTAMP'] - birthdates[x['PID']]).days // 365.25) if birthdates[x['PID']] is not None else -100, axis=1)
        # ages = (concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days // 365.25
        


        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    feature = 'abspos'
    def create(self, concepts):
        abspos = self.config.abspos
        origin_point = datetime(abspos.year, abspos.month, abspos.day)
        # Calculate days since origin point
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.days

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = 'segment'
    def create(self, concepts):
        # Infer NaNs in ADMISSION_ID
        segments = concepts.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0]+1)

        concepts['SEGMENT'] = segments
        return concepts


class BackgroundCreator(BaseCreator):
    feature = 'background'
    def create(self, concepts):
        patients_info_file = glob.glob('patients_info.*', root_dir=self.config.data_dir)[0]
        patients_info = self.read_file(self.config, patients_info_file)

        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.background),
            'CONCEPT': itertools.chain.from_iterable([patients_info[col].tolist() for col in self.config.background])
        }

        for feature in self.config.features:
            if feature != self.feature:
                background[feature.upper()] = 0

        background = pd.DataFrame(background)

        return pd.concat([background, concepts])
        
