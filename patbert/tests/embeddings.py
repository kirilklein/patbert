import string

import torch
from torch import nn

from patbert.features import embeddings

# Define a vocabulary of 10 tokens
vocab_size = 10
# Define the dimensionality of the token embeddings
embedding_dim = 2


# Initialize the token embeddings randomly
embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

# Define some example tokens as input
tokens = torch.tensor([1, 2, 3, 4, 5, 1 ,2, 1, 2])
# let's assume that token 2, 4, 5 are lab tests

# Generate the token embeddings


# Define a simple model that takes the token embeddings as input and 
# predicts a target value
class Model(nn.Module):
    
    def __init__(self):
        # how many layers?
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 20)
        self.F1 = nn.Linear(20, 3)
        self.nF1 = nn.GELU()
        self.F2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.F1(x)	
        x = self.nF1(x)
        x = self.F2(x)
        
        return x
# list all capital english letters


model = Model()
token_embeddings = model.embedding(tokens)
print(token_embeddings.shape)
multiplier = torch.ones(len(tokens))
multiplier[2:6] = 1.4
token_embeddings = token_embeddings * multiplier.reshape(-1, 1)
# Print the generated token embeddings
print(torch.mean(torch.linalg.vector_norm(token_embeddings, axis=1)))
# Assuming that your embedding vectors are stored in a NumPy array called 'embeddings'
norms = torch.linalg.norm(token_embeddings, axis=1, keepdims=True)
# Define some target values for the model to predict
targets = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5   , 0.0, 0.0]
# Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Train the model
"""
for token, target in zip(tokens, targets):
    # zero the parameter gradients
    optimizer.zero_grad()
    # Make predictions using the model
    prediction = model(token)
    # Calculate the loss
    loss = loss_fn(prediction, torch.tensor(target))
    
    # Backpropagate the error and update the model parameters
    loss.backward()
    optimizer.step()
"""
targets = torch.tensor(targets)
for i in range(10):
    # zero the parameter gradients
    optimizer.zero_grad()
    # Make predictions using the model
    predictions = model(tokens)

    # Calculate the loss
    loss = loss_fn(predictions.flatten(), targets)
    
    # Backpropagate the error and update the model parameters
    loss.backward()
    optimizer.step()    
example_x = torch.ones((3, 2))
#print(model.F1(example_x))
#print(model.F1.weight)