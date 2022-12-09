from patbert.models import mlm_plos_pretraining
from os.path import join


def pretraining_test():
        data_file = join("data","tokenized", "simulated.pt")
        vocab_file = join("data","vocabs", "simulated.pt")

        mlm_plos_pretraining.main(data_file=data_file, vocab_file=vocab_file,
                save_path="models/mlm_pretrained/test.pt", 
                epochs=1, 
                batch_size=64, 
                max_len=100,
                config_file="configs\\pretrain_config.json",
                checkpoint_freq=1)