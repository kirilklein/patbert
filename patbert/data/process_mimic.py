import pandas as pd
from os.path import join
from patbert.data import process_utils


class MIMIC3Processor(process_utils.BaseProcessor):
    def __init__(self, cfg) -> None:
        super(MIMIC3Processor, self).__init__()
        self.cfg = cfg
        self.data_path = self.cfg.data_path

    def process_patient_info(self):
        patients = self.load_patients()
        patients = self.remove_birthdates(patients)
        print(patients)

    def remove_birthdates(self, patients, transfers, threshold=100):
        """
            For some patients, the time between birthdate and first admission 
            is unrealistically high (e.g. 100+ years). We drop birthdates for these patients.
        """
        transfers = self.load_transfers()     
        transfers = transfers.loc[transfers['CONEPT']=="THOSPITAL"]
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
        transfers = pd.read_parquet(join(self.data_path,"transfers.parquet"))
        self.convert_to_date(transfers, "TIMESTAMP")
        self.convert_to_date(transfers, "TIMESTAMP_END")

    def convert_to_date(self, df, col):
        df[col] = df[col].dt.date