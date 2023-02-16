import glob
import logging
import os
from collections import defaultdict
from datetime import datetime
from os.path import dirname, join, realpath, split

import hydra
import numpy as np
import pandas as pd
import torch

config_name = "config"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')
log = logging.getLogger(__name__)

@hydra.main(config_name='loading.yaml', config_path=config_path, version_base='1.3')
def process(cfg):
    test=True
    features = defaultdict(list)

    concepts = load_concepts(cfg)
    if test:
        concepts = concepts.sample(n=100000)

    if cfg.get('ages') is not None:
        log.info('ages')
        ages = create_age_features(cfg, concepts)
        concepts['age'] = ages
        features['age'] = []

    if cfg.get('abspos') is not None:
        log.info('abspos')
        abspos = create_abspos_features(cfg, concepts)
        concepts['abspos'] = abspos
        features['abspos'] = []

    if cfg.get('segments') is not None:
        log.info('segments')
        segments = create_segment_features(cfg, concepts)
        concepts['segment'] = segments
        features['segment']  = []

    if cfg.get('demographics') is not None:
        log.info('demographics')
        demographics = create_demographics(cfg)
        concepts = pd.concat((demographics, concepts))
    concepts.groupby('PID', sort=False).apply(lambda patient: add_to_features(features, patient))
    sequence_dir = split(cfg.sequence_file)[0]
    if not os.path.exists(sequence_dir):
        os.makedirs(sequence_dir)
    with open(cfg.sequence_file, 'wb') as f:
        torch.save(features, f)

    return concepts


def load_concepts(cfg):
    # Get all concept files
    path = glob.glob('concept.*', root_dir=cfg.data_dir)

    # Filter out concepts files
    if cfg.get('concepts') is not None:
        path = [p for p in path if p.split('.')[1] in cfg.concepts]
    
    # Load concepts
    concepts = pd.concat([read_file(cfg, p) for p in path])

    return concepts

def create_age_features(cfg: dict, concepts: pd.DataFrame):
    patients_info = read_file(cfg, cfg.patients_info)
    birthdates = patients_info[['PID', 'BIRTHDATE']].set_index('PID', verify_integrity=False).to_dict('index')
    return concepts.apply(lambda patient: calculate_age(patient, birthdates), axis=1)

def create_abspos_features(cfg: dict, concepts: pd.DataFrame):
    year, month, day = cfg.abspos['year'], cfg.abspos['month'], cfg.abspos['day']
    origin_point = datetime(year, month, day)
    return (concepts['TIMESTAMP'] - origin_point).days

def create_segment_features(cfg: dict, concepts: pd.DataFrame):
    mask = concepts['ADMISSION_ID'].isna()
    concepts.loc[mask, 'ADMISSION_ID'] = concepts[mask]['TIMESTAMP'].map(lambda timestamp: f'visit_{timestamp}').values

    segment_dict = {}
    concepts.apply(lambda entry: construct_segment_dict(entry['PID'], entry['ADMISSION_ID'], segment_dict), axis=1)

    return concepts.apply(lambda entry: segment_dict[entry['PID']][entry['ADMISSION_ID']], axis=1)

# TODO: Generalize
def create_demographics(cfg: dict, ages=False, abspos=False, segments=False):
    demo = read_file(cfg, cfg.patients_info)
    background = {
        'PID': demo['PID'].tolist() * 2,
        'CONCEPTS': demo['GENDER'].tolist() + demo['BMI'].tolist()
    }

    # Set optional features to 0 if present
    if ages: background['age'] = 0
    if abspos: background['abspos'] = 0
    if segments: background['segment'] = 0

    demographics = pd.DataFrame(background)
    return demographics


def calculate_age(patient, birthdates):
    if patient['PID'] in birthdates:
        return (patient['TIMESTAMP'] - birthdates[patient['PID']]['BIRTHDATE']).dt.days // 365.25
    else:
        return np.nan

def construct_segment_dict(key, admission, segment_dict):
    if key not in segment_dict:
        segment_dict[key] = {}
    if admission not in segment_dict[key]:
        segment_dict[key][admission] = len(segment_dict[key]) + 1

def add_to_features(features, patient):
    for column in patient:
        features[column].append(patient[column].tolist())


def read_file(cfg, file_path, n_rows=None, test=False):
    file_path = join(cfg.data_dir,file_path)
    file_type = file_path.split(".")[-1]
    if file_type == 'csv':
        return pd.read_csv(file_path, nrows=n_rows)
    elif file_type == 'parquet':
        return pd.read_parquet(file_path)

if __name__ == '__main__':
    process()