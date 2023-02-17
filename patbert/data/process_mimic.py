import pandas as pd
from os.path import join
from patbert.data import process_utils


class MIMIC3Processor(process_utils.BaseProcessor):
    def __init__(self, cfg, test=False) -> None:
        super(MIMIC3Processor, self).__init__(cfg)
        self.cfg = cfg
        self.test = test
        self.data_path = self.cfg.data_path

    def convert_to_date(self, df, col):
        df[col] = df[col].dt.date
        

    def __call__(self):
        PatientProcessor(self.cfg, self.test)()

class PatientProcessor(MIMIC3Processor):
    def __init__(self, cfg, test) -> None:
        super(PatientProcessor, self).__init__(cfg, test)

    def __call__(self):
        patients = self.load_patients()
        patients = self.remove_birthdates(patients)
        
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
