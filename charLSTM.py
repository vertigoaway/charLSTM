import torch
from torch import nn
from torch.utils.data import DataLoader
import csv
import numpy.typing as npt
import numpy as np

import trinketbox.ai.utils.tokenDataset as integerDataset

from griot import char
from griot import tool as griotTools
#0 is null
#1 is end of sent
vocab = char.Vocab()
vocab.addCharacters(list('abcdefghijklmnopqrstuvwxyz -,.:;\''))


### LSTM Architecture Parameters
inSize : int = 512         # Context window
outSize : int = 1          # How many chars to predict
embedding_dim : int = 384  # Embedding dimension for vocabulary
hidden_size : int = 768    # Hidden size for each LSTM layer
num_layers : int = 2       # Number of LSTM layers
dropout : float = 0.2      # Dropout for regularization between LSTM layers
device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPath = 'model.pth'
### Training params
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam
learning_rate : float = 5e-4
batch_size : int = 20
epochs : int = 10
trainingData = "data.csv"


### Load data
with open(trainingData, "r") as csvfile:
    readout = list(csv.reader(csvfile))[1:]
    out = []
    for r in readout:
        if len(r[3]) > 3:
            out.append(r[-1].strip().lower())
readout = out


### Begin tokenizing data
x : list[int] = griotTools.flattenTokenizedLines(vocab.tokenizeLines(out))


train_dataSet = integerDataset.lazyTextDataset(inSize=inSize,outSize=outSize,
                                 tokenizedData=x[0:len(x)//2],
                                 vocSize=len(vocab))
test_dataSet = integerDataset.lazyTextDataset(inSize=inSize,outSize=outSize,
                                tokenizedData=x[len(x)//2:],
                                vocSize=len(vocab))
train_dataloader = DataLoader(train_dataSet, batch_size=batch_size, 
                              shuffle=True,
                              num_workers=4)
test_dataloader = DataLoader(test_dataSet, batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)



###
class NeuralNetwork(nn.Module):
    def __init__(self, vocSize, inSize, outSize, 
                 embedding_dim=128, 
                 hidden_size=256, 
                 num_layers=2, 
                 dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocSize, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.layerNorm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, outSize * vocSize)
        self.outSize = outSize
        self.vocSize = vocSize
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.layerNorm(x)
        logits = self.linear(x)  # (B, V)
        return logits




model = NeuralNetwork(vocSize=len(vocab), inSize=inSize, outSize=outSize, 
                      embedding_dim=embedding_dim, hidden_size=hidden_size, 
                      num_layers=num_layers, dropout=dropout).to(device)
try:
    print('loading last save')
    model.load_state_dict(torch.load(modelPath))
except FileNotFoundError:
    print('loading failed, starting from scratch')
print(model)






