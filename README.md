# Patient Embeddings with BERT
We use ICD10 codes, medication and procedures instead of words and visits instead of sentences. 
As pretraining tasks MLM and prolonged length of stay in the hospital (>7 days) are employed. The task will be fine-tuned for hospitalization/ICU prediction of COVID patients.

## Reproducing

Run the following steps:

    If you want an example dataset:  
      python data\generate.py <num_patients> <save_name> 
    python data\tokenize_example_data.py <input_data_file>  
    python models\mlm_plos_pretraining.py <path_to_tokenized_data> <path_to_vocab_file> <path_to_save_model> <epochs> 
