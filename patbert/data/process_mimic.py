import datetime
import os
from os.path import join
from shutil import copyfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hydra import utils as hydra_utils
from pyarrow.parquet import ParquetFile

from patbert.data import process_base, utils


class MIMIC3Processor(process_base.BaseProcessor):
    def __init__(self, cfg, test=False) -> None:
        super(MIMIC3Processor, self).__init__(cfg)
        self.cfg = cfg
        self.test = test
        self.data_path = self.cfg.data_path
        self.concept = None
        
    @utils.timing_function
    def __call__(self):
        print("Processing MIMIC-III data")
        for category in self.cfg.include:
            print(f" :{category}")
            processor = globals()[f"{category}Processor"](self.cfg, self.test)
            if processor.concept in self.cfg.no_processing:
                self.copy_table_to_processed_folder(processor.concept)
            else:
                processor()
    def load(self):
        if not self.test:
            return pd.read_parquet(join(self.data_path,f"concept.{self.concept}.parquet"))
        else:
            pf = ParquetFile(join(self.data_path,f"concept.{self.concept}.parquet"))
            batch = next(pf.iter_batches(batch_size = int(1e5))) 
            return pa.Table.from_batches([batch]).to_pandas() 

    def load_transfers(self):
        transfers = pd.read_parquet(join(self.data_path,"concept.transfer.parquet"))
        self.convert_to_date(transfers, "TIMESTAMP")
        self.convert_to_date(transfers, "TIMESTAMP_END")
        return transfers
    
    def copy_table_to_processed_folder(self, concept):
        # copy file from data_path to new folder
        src_path = join(self.data_path,f"concept.{concept}.parquet")
        dest_path = join(os.getcwd(), f"concept.{concept}.parquet")
        copyfile(src_path, dest_path)
        

class PatientInfoProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(PatientInfoProcessor, self).__init__(cfg, test)
        self.conf = self.cfg.patients_info
    @utils.timing_function
    def __call__(self):
        patients = self.load_patients()
        patients = self.remove_birthdates(patients)
        hydra_utils.call(self.conf.group_rare_values, df=patients)
        pq.write_table(pa.Table.from_pandas(patients), join(os.getcwd(), f'patients_info.parquet'))

    def remove_birthdates(self, patients, threshold=110):
        """
            For some patients, the time between birthdate and first admission 
            is unrealistically high (e.g. 110+ years). We drop birthdates for these patients.
        """
        transfers = self.load_transfers()     
        transfers = transfers.loc[transfers['CONCEPT']=="THOSPITAL"]
        transfers = pd.merge(transfers, patients[["PID", "BIRTHDATE"]], on="PID", how="left")
        transfers["admission_age"] = (transfers.TIMESTAMP - transfers.BIRTHDATE).map(lambda x: x.days / 365.25)
        change_bd = transfers[transfers.admission_age > threshold].PID.unique()
        print("  :: Change birthdates for", len(change_bd), "patients , by adding 200 years.")
        patients.loc[patients.PID.isin(change_bd), "BIRTHDATE"] = patients.loc[patients.PID.isin(change_bd), "BIRTHDATE"] + datetime.timedelta(days=365.25*200)
        return patients

    def load_patients(self,):
        patients = pd.read_parquet(join(self.data_path,"patients_info.parquet"))
        self.convert_to_date(patients, "BIRTHDATE")
        self.convert_to_date(patients, "DEATHDATE")
        return patients

    

class TransfersProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(TransfersProcessor, self).__init__(cfg, test)
        self.conf = self.cfg.transfer
        self.concept = "transfer"
    @utils.timing_function
    def __call__(self):
        transfers = self.load()
        transfers = self.separate_start_end(transfers)
        transfers = self.append_hospital_admission_type(transfers)
        transfers = self.append_hospital_discharge_location(transfers)
        transfers = self.write_concept_to_parquet(transfers)
    
    @staticmethod
    def separate_start_end(transfers):
        """Separate transfers into start and end events"""
        transfers_start = transfers.copy().drop(columns=['TIMESTAMP_END'])
        transfers_end = transfers.copy().drop(columns=['TIMESTAMP'])
        transfers_start['CONCEPT'] = transfers_start['CONCEPT'] + '_START'
        transfers_end['CONCEPT'] = transfers_end['CONCEPT'] + '_END'
        transfers_end = transfers_end.rename(columns={'TIMESTAMP_END': 'TIMESTAMP'})
        transfers = pd.concat([transfers_start, transfers_end]).reset_index(drop=True)
        return transfers
    @staticmethod
    def append_hospital_admission_type(transfers):
        """Append admission type to THOSPITAL_START"""
        start_mask = (transfers.CONCEPT=='THOSPITAL_START')
        transfers.loc[start_mask, 'CONCEPT'] = transfers.loc[start_mask, 'CONCEPT'] \
            + '_' + transfers.loc[start_mask, 'ADMISSION_TYPE']
        return transfers
    @staticmethod
    def append_hospital_discharge_location(transfers):
        """Append discharge location to THOSPITAL_END"""
        end_mask = (transfers.CONCEPT=='THOSPITAL_END')
        transfers.loc[end_mask, 'CONCEPT'] = transfers.loc[end_mask, 'CONCEPT'] \
            + '_' + transfers.loc[end_mask, 'DISCHARGE_LOCATION']
        return transfers

class WeightsProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(WeightsProcessor, self).__init__(cfg, test)
        self.conf = self.cfg.weight
        self.concept = "weight"
    @utils.timing_function
    def __call__(self):
        weights = self.load()
        if self.conf.drop_constant:
            weights = self.drop_constant_weight(weights)
        self.write_concept_to_parquet(weights)
    
    @staticmethod
    def drop_constant_weight(weights):
        """There are many repeating weight measurements with unchanged weight. 
        Keep only the first one, or when weight changes"""
        weights['weight_diff'] = weights.groupby('PID').VALUE.diff()
        weights = weights[weights.weight_diff!=0]
        weights = weights.drop(columns=['weight_diff'])
        return weights

# EventProcessor is a base class for all event tables (diagnoses, procedures, etc.)
class EventProcessor(MIMIC3Processor):
    def __init__(self, cfg, test, concept) -> None:
        super(EventProcessor, self).__init__(cfg, test)
        self.concept = concept
        self.conf = self.cfg[self.concept]

    def __call__(self):
        """Call which is the same for all concept tables"""
        events = self.load()
        events = self.drop_missing_timestamps(events)
        events = hydra_utils.call(self.conf.group_rare_values, df=events, cols=['CONCEPT'])
        return events

class DiagnosesProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(DiagnosesProcessor, self).__init__(cfg, test, "diag")
    @utils.timing_function
    def __call__(self):
        diagnoses = super().__call__()
        self.write_concept_to_parquet(diagnoses)

class ProceduresProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(ProceduresProcessor, self).__init__(cfg, test, "pro")
    @utils.timing_function
    def __call__(self):
        procedures = super().__call__()
        self.write_concept_to_parquet(procedures)

class MedicationsProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(MedicationsProcessor, self).__init__(cfg, test, "med")
    @utils.timing_function
    def __call__(self):
        medications = super().__call__()
        self.write_concept_to_parquet(medications)


class LabEventsProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(LabEventsProcessor, self).__init__(cfg, test, "lab")
    @utils.timing_function
    def __call__(self):
        lab_events = super().__call__()
        self.write_concept_to_parquet(lab_events)

class MicrobiologyProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(MicrobiologyProcessor, self).__init__(cfg, test, "microbio")
    @utils.timing_function
    def __call__(self):
        microbiology_events = super().__call__()
        self.write_concept_to_parquet(microbiology_events)

class ChartEventsProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(ChartEventsProcessor, self).__init__(cfg, test, "chartevent")
    @utils.timing_function
    def __call__(self):
        chartevents = super().__call__()
        self.write_concept_to_parquet(chartevents)

class ChartEventsMainProcessor(EventProcessor):
    def __init__(self, cfg, test) -> None:
        super(ChartEventsMainProcessor, self).__init__(cfg, test, "chartevent_main")
    @utils.timing_function
    def __call__(self):
        chartevents = super().__call__()
        self.write_concept_to_parquet(chartevents)

# patients = hydra_utils.call(self.cfg.values_processing, cfg=self.cfg, df=patients)