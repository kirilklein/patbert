import numpy as np
import pickle as pkl
import string 
import typer
from datetime import datetime


class DataGenerator(super):
    def __init__(self, num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, min_los, 
        max_los, num_atc_codes, num_icd_codes, num_lab_test=500, age_lower_bound=0, seed=42,
        start_date=datetime(2010, 1, 1)):
        """
        Simulates data as lists:
            [pid, los_ls, all_visit_codes, visit_nums]
        min_los, max_los: Length os stay in the hospital,
        num_codes: total number of ICD10 codes to generate
        """
        self.num_patients = num_patients
        self.min_num_codes_per_visit = min_num_codes_per_visit
        self.max_num_codes_per_visit = max_num_codes_per_visit
        self.min_num_visits = min_num_visits
        self.max_num_visits = max_num_visits
        self.min_los = min_los
        self.max_los = max_los
        self.num_icd_codes = num_icd_codes
        self.num_atc_codes = num_atc_codes
        self.age_lower_bound = age_lower_bound
        self.num_lab_tests = num_lab_test
        self.start_date = start_date
        self.rng = np.random.default_rng(seed)
    
    # generate atc codes
    def generate_patient_history(self, pid):
        """Generates a dictionary which contains sex, ages, length of stay, codes, lab tests, lab tests visits"""
        num_visits = self.rng.integers(self.min_num_visits, self.max_num_visits)
        num_codes_per_visit_ls = self.rng.integers(self.min_num_codes_per_visit, 
            self.max_num_codes_per_visit, 
            size=num_visits) # should icd and atc vectors point in different directions
        los = self.rng.integers(self.min_los, self.max_los, size=num_visits)\
            .tolist()
        los = np.repeat(los, num_codes_per_visit_ls).tolist()
        icd_codes = self.generate_randomICD10_codes(self.num_icd_codes)
        atc_codes = self.generate_randomATC_codes(self.num_atc_codes)
        lab_tests = self.generate_lab_tests(self.num_lab_tests)
        codes = icd_codes + atc_codes + lab_tests
        values = [1]*(len(icd_codes)+len(atc_codes)) + self.rng.normal(size=len(lab_tests)).tolist()
        modalities = ['ICD10']*len(icd_codes) + ['ATC']*len(atc_codes) + ['LAB']*len(lab_tests)
        idx = self.rng.choice(np.arange(len(codes)), np.sum(num_codes_per_visit_ls), replace=True)
        codes = np.array(codes)[idx].tolist()
        modalities = np.array(modalities)[idx].tolist()
        values = np.array(values)[idx].tolist()
        visit_nums = np.arange(1, num_visits+1) # should start with 1!
        visit_nums = np.repeat(visit_nums, num_codes_per_visit_ls).tolist()
        
        birthdate = self.generate_birthdate()
        ages = self.generate_ages(num_visits, birthdate) # pass age as days or rounded years?
        absolute_position = self.generate_absolute_position(ages, birthdate) # in days
        ages = np.repeat(ages, num_codes_per_visit_ls).tolist()
        absolute_position = np.repeat(absolute_position, num_codes_per_visit_ls).tolist()

        patient_dic = {
            'pid':pid,
            'birthdate': birthdate,
            'sex':self.generate_sex(),
            'codes':codes,
            'ages':ages,
            'los':los,
            'visits':visit_nums,
            'absolute_position':absolute_position,
            'modalities':modalities,
            'values':values
        }
        return patient_dic


    def generate_ages(self, num_visits, birthdate):
        ages = []
        age_lower_bound = self.age_lower_bound
        age_upper_bound = int(((datetime.today() - birthdate).days)/365)
        random_age = self.rng.integers(age_lower_bound, age_upper_bound)
        for _ in range(num_visits):
            if random_age > age_upper_bound:
                random_age = age_upper_bound
            ages.append(random_age)
            age_lower_bound = random_age 
            random_age = self.rng.poisson(2, 1)[0] + random_age
        return ages
        
    def generate_absolute_position(self, ages, birthdate):
        absolute_positions = []
        birthdate_difference = self.start_date - birthdate
        for age in ages:
            days_since_start = age*365-birthdate_difference.days
            days_since_start += self.rng.poisson(2,1)[0]
            absolute_positions.append(days_since_start)
        return absolute_positions

    def generate_sex(self):
        return self.rng.binomial(1, 0.5)

    def generate_birthdate(self):
        year = self.rng.integers(1900, 1980)
        month = self.rng.integers(1, 12)
        day = self.rng.integers(1, 28)
        return datetime(year, month, day)

    def generate_randomICD10_codes(self, n):
        letters = self.rng.choice([char for char in string.ascii_uppercase], 
            size=n, replace=True)
        numbers_category = self.rng.choice(np.arange(100), size=n, replace=True)
        numbers_subcategory = self.rng.choice(np.arange(1000), size=n, replace=True)
        lengths = self.rng.integers(low=1, high=4, size=n)
        codes = [letter + str(number_category).zfill(2) + str(number_subcategory).zfill(3)[:length] for \
            letter, number_category, number_subcategory, length \
                in zip(letters, numbers_category, numbers_subcategory, lengths)]
        return codes

    def generate_randomATC_codes(self, n):
        letters = np.random.choice([char for char in string.ascii_uppercase], 
            size=2*n, replace=True)
        numbers = np.random.choice(np.arange(100), size=2*n, replace=True)
        codes = [letter0 + str(number0).zfill(2) + letter1 + str(number1).zfill(2) for \
            letter0, number0, letter1, number1 in zip(letters[:n], numbers[:n], letters[n:], numbers[n:])]
        return codes
    
    def generate_lab_tests(self, num_lab_tests):
        lab_tests = []
        for _ in range(num_lab_tests):
            lab_tests.append(('LT'+str(self.rng.integers(self.num_lab_tests))))
        return lab_tests

    def simulate_data(self):
        for pid in range(self.num_patients):
            yield self.generate_patient_history('p_'+str(pid))

def main(num_patients : int = typer.Argument(...), 
        save_name: str = typer.Argument(..., 
        help="name of the file to save the data to, should end with .pkl"),
        min_num_visits: int = typer.Option(2),
        max_num_visits: int = typer.Option(10),
        min_num_codes_per_visit: int = typer.Option(1),
        max_num_codes_per_visit: int = typer.Option(5),
        min_los: int = typer.Option(1),
        max_los: int = typer.Option(30),
        num_atc_codes: int = typer.Option(1000),
        num_icd_codes: int = typer.Option(1000),
        num_lab_tests: int = typer.Option(1000)):
    generator = DataGenerator(num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, 
        min_los, max_los, num_atc_codes, num_icd_codes, num_lab_tests)
    with open(save_name, 'wb') as f:
        pkl.dump([hist for hist in generator.simulate_data()], f)
    #print([hist for hist in generator.simulate_data()])
if __name__ == '__main__':
    typer.run(main)