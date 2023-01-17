import json
import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining

from patbert.common import pytorch
from patbert.models.trainers import CustomPreTrainer



class Encoder(CustomPreTrainer):
    def __init__(self, dataset, model_dir, pat_ids,
                from_checkpoint=False, batch_size=128):
        self.model_dir = model_dir
        with open(join(self.model_dir, 'config.json'), 'r') as f:
            config_dic = json.load(f)
        self.config = BertConfig(**config_dic)
        super().__init__(train_dataset=dataset, val_dataset=None, model=None,
                        epochs=None, batch_size=batch_size, model_dir=model_dir,   
                        from_checkpoint=from_checkpoint, config=self.config)
        if not from_checkpoint:
            print(f"Load saved model from {model_dir}")
            self.model = torch.load(join(model_dir, 'model.pt'))
        else:
            model = BertForPreTraining(self.config)
            self.model = self.load_from_checkpoint(model, None)
        self.pat_ids = pat_ids
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.from_checkpoint:
            self.model = self.load_from_checkpoint(self.model, None)
        self.model.to(device) # type: ignore # and move our model over to the selected device
        self.model.eval()  # type: ignore
        loader = torch.utils.data.DataLoader(self.train_dataset,  # type: ignore
                                    batch_size=self.batch_size, shuffle=False)  
        loop = tqdm(loader, leave=True)                        
        pat_vecs = []
        for batch in loop:
            # put all tensore batches required for training
            batch = pytorch.batch_to_device(batch, device)
            # get embeddings
            embedding_output = self.embeddings(batch['codes'], batch['segments'])
            # process
            outputs = self.model(inputs_embeds=embedding_output,   # type: ignore
                        attention_mask=batch['attention_mask'],  
                        labels=batch['labels'],
                        next_sentence_label=batch['plos'], 
                        output_hidden_states=True) # type: ignore                
            loop.set_description(f"Inference")
            
            for i, hidden_state in enumerate(outputs.hidden_states[-1]):
                itemindex = np.where(np.array(batch['codes'][i]) == 0)
                if len(itemindex[0]) > 0:
                    length = itemindex[0][0] # take only non-padded tokens
                    pat_vec = hidden_state[:length,:].mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
                else:
                    pat_vec = hidden_state.mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
        pat_vecs = np.stack(pat_vecs, axis=0)
        assert len(pat_vecs) == len(self.pat_ids)  # type: ignore
        if not os.path.exists(join(self.model_dir, 'encodings')):
            os.makedirs(join(self.model_dir, 'encodings'))
        np.savez(join(self.model_dir, 'encodings', 'encodings.npz'), 
                **{'pat_ids':self.pat_ids, 'pat_vecs':pat_vecs})
        return self.pat_ids, pat_vecs

class FCLayer(nn.Module):
    """A fully connected layer with GELU activation to train on top of static embeddings"""
    def __init__(self, input_size, output_size, weight=None, bias=None):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        if not(isinstance(weight, type(None))):
            self.fc.weight = torch.nn.Parameter(weight)
        if not(isinstance(bias, type(None))):
            self.fc.bias = torch.nn.Parameter(bias)
        self.act = nn.GELU()

    def forward(self, x):
        # Add the GELU nonlinearity after the linear layer
        x = self.fc(x)
        x = self.act(x)
        return x

class ModelFC(nn.Module):
    """Insert a fully connected layer with nonlinearity on top of static embeddings before feeding into model"""
    def __init__(self, model, config):
        super(ModelFC, self).__init__()
        self.model = model
        self.fc1 = FCLayer(config.hidden_size, config.hidden_size)

    def forward(self, x):
        # Pass the input tensor through the FC layer before passing it through the BERT model
        x = self.fc1(x)
        x = self.model(x)
        return x


class Attention(Encoder):
    def __init__(self, dataset, model_dir, pat_ids, 
                from_checkpoint=False, batch_size=128):
        super().__init__(dataset, model_dir, pat_ids, from_checkpoint, batch_size)
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.from_checkpoint:
            self.model = self.load_from_checkpoint(self.model, None)
        self.model.to(device) # type: ignore # and move our model over to the selected device
        self.model.eval()  # type: ignore
        loader = torch.utils.data.DataLoader(self.train_dataset,  # type: ignore
                                    batch_size=self.batch_size, shuffle=False)  
        loop = tqdm(loader, leave=True)                        
        pat_vecs = []
        for batch in loop:
            # put all tensore batches required for training
            batch = pytorch.batch_to_device(batch, device)
            # get embeddings
            embedding_output = self.embeddings(batch['codes'], batch['segments'])
            # process
            # TODO: Use output vectors to compute loss
            outputs = self.model(inputs_embeds=embedding_output,   # type: ignore
                        attention_mask=batch['attention_mask'],  
                        labels=batch['labels'],
                        next_sentence_label=batch['plos'], 
                        output_hidden_states=True) # type: ignore                
            loop.set_description(f"Inference")
            
            for i, hidden_state in enumerate(outputs.hidden_states[-1]):
                itemindex = np.where(np.array(batch['codes'][i]) == 0)
                if len(itemindex[0]) > 0:
                    length = itemindex[0][0] # take only non-padded tokens
                    pat_vec = hidden_state[:length,:].mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
                else:
                    pat_vec = hidden_state.mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
        pat_vecs = np.stack(pat_vecs, axis=0)
        assert len(pat_vecs) == len(self.pat_ids)  # type: ignore
        if not os.path.exists(join(self.model_dir, 'encodings')):
            os.makedirs(join(self.model_dir, 'encodings'))
        np.savez(join(self.model_dir, 'encodings', 'encodings.npz'), 
                **{'pat_ids':self.pat_ids, 'pat_vecs':pat_vecs})
        return self.pat_ids, pat_vecs