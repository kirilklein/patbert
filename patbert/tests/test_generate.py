from patbert.data import generate
import numpy as np

def test_generate(num_patients=10,
        min_num_visits=2,
        max_num_visits=5,
        min_num_codes_per_visit=1,
        max_num_codes_per_visit=5,
        min_los=1,
        max_los=30,
        num_atc_codes=1000, num_icd_codes=1000, num_lab_tests=1000):
    generator = generate.DataGenerator(
        num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, 
        min_los, max_los, num_atc_codes=num_atc_codes,num_icd_codes=num_icd_codes,num_lab_test=num_lab_tests
    )
    data = [pat for pat in generator.simulate_data()]
    assert len(data) == num_patients
    #assert np.min(np.array([min(d['los']) for d in data])) >= min_los
    #assert np.max(np.array([max(d['los']) for d in data])) <= max_los
    assert np.min(np.array([min(d['visits']) for d in data])) <= min_num_visits
    assert np.max(np.array([max(d['visits']) for d in data])) <= max_num_visits
    for pat in data:
        assert len(pat['codes']) == len(pat['visits']) == len(pat['ages'])

