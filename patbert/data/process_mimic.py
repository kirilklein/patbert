from os.path import join

import pandas as pd
import pyarrow as pa
from hydra import utils as hydra_utils
from pyarrow.parquet import ParquetFile

from patbert.data import process_base


class MIMIC3Processor(process_base.BaseProcessor):
    def __init__(self, cfg, test=False) -> None:
        super(MIMIC3Processor, self).__init__(cfg)
        self.cfg = cfg
        self.test = test
        self.data_path = self.cfg.data_path
        self.concept = None

    def __call__(self):
        print("Processing MIMIC-III data")
        for category in self.cfg.include:
            print(f":{category}")
            processor = globals()[f"{category}Processor"](self.cfg, self.test)
            processor()
    
    def load(self):
        if not self.test:
            return pd.read_parquet(join(self.data_path,f"concept.{self.concept}.parquet"))
        else:
            pf = ParquetFile(join(self.data_path,f"concept.{self.concept}.parquet"))
            batch = next(pf.iter_batches(batch_size = int(1e5))) 
            return pa.Table.from_batches([batch]).to_pandas() 


class PatientInfoProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(PatientInfoProcessor, self).__init__(cfg, test)
        self.conf = self.cfg.patients_info

    def __call__(self):
        patients = self.load_patients()
        patients = self.remove_birthdates(patients)
        if self.conf.group_rare_values:
            for col in self.conf.group_rare_values_cols:
                patients = self.group_rare_values(patients, col)

    def remove_birthdates(self, patients, threshold=110):
        """
            For some patients, the time between birthdate and first admission 
            is unrealistically high (e.g. 110+ years). We drop birthdates for these patients.
        """
        transfers = self.load_transfers()     
        transfers = transfers.loc[transfers['CONCEPT']=="THOSPITAL"]
        transfers = pd.merge(transfers, patients[["PID", "BIRTHDATE"]], on="PID", how="left")
        transfers["admission_age"] = (transfers.TIMESTAMP - transfers.BIRTHDATE).map(lambda x: x.days / 365.25)
        remove_pids = transfers[transfers.admission_age > threshold].PID.unique()
        patients.loc[patients.PID.isin(remove_pids), "BIRTHDATE"] = pd.NaT
        return patients

    def load_patients(self,):
        patients = pd.read_parquet(join(self.data_path,"patients_info.parquet"))
        self.convert_to_date(patients, "BIRTHDATE")
        self.convert_to_date(patients, "DEATHDATE")
        return patients

    def load_transfers(self):
        transfers = pd.read_parquet(join(self.data_path,"concept.transfer.parquet"))
        self.convert_to_date(transfers, "TIMESTAMP")
        self.convert_to_date(transfers, "TIMESTAMP_END")
        return transfers

class TransfersProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(TransfersProcessor, self).__init__(cfg, test)
        self.conf = self.cfg.transfers
        self.concept = "transfer"

    def __call__(self):
        transfers = self.load()
        # we will separate THOSPITAL in categores based on ADMIT_TYPE
        transfers.loc[transfers.CONCEPT=='THOSPITAL', 'CONCEPT'] = transfers.CONCEPT + '_' + transfers.ADMISSION_TYPE
        # transfers =
    
    @staticmethod
    def separate_start_end(transfers):
        """Separate transfers into start and end events"""
        transfers_start = transfers.copy()
        transfers_end = transfers.copy()
        transfers_start['CONCEPT'] = transfers_start['CONCEPT'] + '_START'
        transfers_end['CONCEPT'] = transfers_end['CONCEPT'] + '_END'
        transfers_end = transfers_end.rename(columns={'TIMESTAMP_END': 'TIMESTAMP'})
        transfers = pd.concat([transfers_start, transfers_end])
        return transfers

class WeightsProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(WeightsProcessor, self).__init__(cfg, test)
        self.conf = self.cfg.weight
        self.concept = "weight"

    def __call__(self):
        weights = self.load()
        if self.conf.drop_constant:
            weights = self.drop_constant_weight(weights)
    
    @staticmethod
    def drop_constant_weight(weights):
        """There are many repeating weight measurements with unchanged weight. 
        Keep only the first one, or when weight changes"""
        weights['weight_diff'] = weights.groupby('PID').VALUE.diff()
        weights = weights[weights.weight_diff!=0]
        weights = weights.drop(columns=['weight_diff'])
        return weights

class EventProcessor(MIMIC3Processor):
    def __init__(self, cfg, test, concept) -> None:
        super(EventProcessor, self).__init__(cfg, test)
        self.concept = concept
        self.conf = self.cfg[self.concept]

    def __call__(self):
        df = self.load()
        if self.conf.group_rare_values:
            df = self.group_rare_values(df, 'CONCEPT', rare_threshold=self.conf.rare_threshold)

class DiagnosesProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(DiagnosesProcessor, self).__init__(cfg, test, "diag")
    
    def __call__(self):
        diagnoses = super().__call__()

class ProceduresProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(ProceduresProcessor, self).__init__(cfg, test, "pro")
    
    def __call__(self):
        procedures = super().__call__()

class MedicationsProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(MedicationsProcessor, self).__init__(cfg, test, "med")

    def __call__(self):
        medications = super().__call__()

class LabEventsProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(LabEventsProcessor, self).__init__(cfg, test, "lab")

    def __call__(self):
        lab_events = super().__call__()

class ChartEventsProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(ChartEventsProcessor, self).__init__(cfg, test, "chartvent")

    def __call__(self):
        chartevents = super().__call__()

# patients = hydra_utils.call(self.cfg.values_processing, cfg=self.cfg, df=patients)