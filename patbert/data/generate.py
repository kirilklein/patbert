import numpy as np
import pickle as pkl
import string 
import typer


class DataGenerator(super):
    def __init__(self, num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, min_los, 
        max_los, num_codes, age_lower_bound=0, age_upper_bound=100,
        num_different_lab_test=100, min_num_lab_tests=0, max_num_lab_tests=100):
        """
        Simulates data as lists:
            [pid, los_ls, all_visit_codes, visit_nums]
        min_los, max_los: Length os stay in the hospital,
        num_codes: total number of ICD10 codes to generate
        """
        self.num_patients = num_patients
        self.min_num_codes_per_visit = min_num_codes_per_visit
        self.max_num_codes_per_visit = max_num_codes_per_visit
        self.min_num_lab_tests = min_num_lab_tests
        self.max_nun_lab_tests = max_num_lab_tests
        self.min_num_visits = min_num_visits
        self.max_num_visits = max_num_visits
        self.min_los = min_los
        self.max_los = max_los
        self.num_codes = num_codes
        self.age_lower_bound = age_lower_bound
        self.age_upper_bound = age_upper_bound
        self.num_different_lab_test = num_different_lab_test
        self.rng = np.radom.default_rng()

    def generate_ICD10_history(self, pid):
        """Generates a list which contains the ICD10 codes (medbert format)"""
        codes = self.generate_randomICD10_codes(self.num_codes)
        num_visits = np.random.randint(self.min_num_visits, self.max_num_visits)
        num_codes_per_visit_ls = np.random.randint(self.min_num_codes_per_visit, 
            self.max_num_codes_per_visit, 
            size=num_visits)
        los_ls = np.random.randint(self.min_los, self.max_los, size=num_visits)\
            .tolist()
        all_visit_codes = np.random.choice(codes, 
            size=np.sum(num_codes_per_visit_ls), replace=True).tolist()
        visit_nums = np.arange(1, num_visits+1) # should start with 1!
        visit_nums = np.repeat(visit_nums, num_codes_per_visit_ls).tolist()
        return [pid, los_ls, all_visit_codes, visit_nums]
    
    # generate atc codes
    def generate_patient_history(self, pid):
        """Generates a dictionary which contains sex, ages, length of stay, codes, lab tests, lab tests visits"""
        num_visits = self.rng.randint(self.min_num_visits, self.max_num_visits)
        num_lab_tests = self.rng.randint(self.min_num_lab_tests, self.max_nun_lab_tests)
        num_codes_per_visit_ls = np.random.randint(self.min_num_codes_per_visit, 
            self.max_num_codes_per_visit, 
            size=num_visits) # should icd and atc vectors point in different directions
        ages = self.generate_ages(num_visits) 
        
        icd_codes = self.generate_randomICD10_codes(self.num_codes)
        atc_codes = self.generate_randomATC_codes(self.num_codes)
        codes = icd_codes + atc_codes
        codes = np.random.choice(codes, 
            size=np.sum(num_codes_per_visit_ls), replace=True).tolist()
        lab_tests = self.generate_lab_tests(num_lab_tests)        
        lab_tests_visits = self.generate_lab_tests_visits(num_lab_tests)
        ages = self.generate_ages_in_days(self) # pass age as days or rounded years?
        # simulate random increasing integers
        # TODO: finish here
        patient_dic = {
            'pid':pid,
            'sex':self.generate_sex(),
            'ages':ages,
            'los':los,
            'codes_segments':codes_visits,
            'codes':codes,
            'lab_tests':lab_tests,
            'lab_tests_segments':lab_tests_visits
        }
        return patient_dic

    def generate_lab_tests(self, num_lab_tests):
        lab_tests = []
        for _ in range(num_lab_tests):
            lab_tests.append(('test'+str(self.rng.randint(self.num_different_lab_test))), self.rng.normal())
        return lab_tests
    def generate_ages(self, num_visits):
        ages = []
        age_lower_bound = self.age_lower_bound
        for i in range(num_visits):
            random_age = self.rng.randint(age_lower_bound, self.age_upper_bound)
            ages.append(random_age)
            age_lower_bound = random_age + 1
        
        return ages

    def generate_sex(self):
        return self.rng.binomial(1, 0.5)

    def generate_randomICD10_codes(self, n):
        letters = np.random.choice([char for char in string.ascii_uppercase], 
            size=n, replace=True)
        numbers = np.random.choice(np.arange(1000), size=n, replace=True)
        codes = [letter + str(number).zfill(3)[:2] + '.' + str(number)[-1] for \
            letter, number in zip(letters, numbers)]
        return codes
    def generate_randomATC_codes(self, n):
        letters = np.random.choice([char for char in string.ascii_uppercase], 
            size=2*n, replace=True)
        numbers = np.random.choice(np.arange(100), size=2*n, replace=True)
        codes = [letter0 + str(number0).zfill(2) + letter1 + str(number1).zfill(2) for \
            letter0, number0, letter1, number1 in zip(letters[:n], numbers[:n], letters[n:], numbers[n:])]
        return codes

    def simulate_data(self):
        for pid in range(self.num_patients):
            yield self.generate_ICD10_history('p_'+str(pid))

def main(num_patients : int = typer.Argument(...), 
        save_name: str = typer.Argument(..., 
        help="name of the file to save the data to, should end with .pkl"),
        min_num_visits: int = typer.Option(2),
        max_num_visits: int = typer.Option(10),
        min_num_codes_per_visit: int = typer.Option(1),
        max_num_codes_per_visit: int = typer.Option(5),
        min_los: int = typer.Option(1),
        max_los: int = typer.Option(30),
        num_codes: int = typer.Option(1000)):
    generator = DataGenerator(num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, 
        min_los, max_los, num_codes)
    with open(save_name, 'wb') as f:
        pkl.dump([hist for hist in generator.simulate_data()], f)

if __name__ == '__main__':
    typer.run(main)