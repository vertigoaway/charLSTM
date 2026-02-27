import torch
from torch import nn
from torch.utils.data import DataLoader
import csv
import numpy.typing as npt
import numpy as np

import trinketbox.ai.utils.NNLoops as loops
import trinketbox.ai.utils.tokenDataset as integerDataset
import trinketbox.ai.utils.charTokenizer as cT
import trinketbox.ai.utils.outProcessing as post



device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#0 is null
#1 is end of sent
voc : dict[str,int]= {'�':0,chr(10):1,'-':2,'_':3,'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,
       'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,
       'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,
       't':23,'u':24,'v':25,'w':26,'x':27,'y':28,
       'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,
       '\"':35,':':36,';':37,'1':38,'2':39,'3':40,
       '4':41,'5':42,'6':43,'7':44,'8':45,'9':46,
       '0':47,}
cov = {i: s for s, i in voc.items()}

vocSize : int = len(voc) #acct for special toks

### LSTM Architecture Parameters
inSize : int = 512
outSize : int = 1
embedding_dim : int = 384  # Embedding dimension for vocabulary
hidden_size : int = 768    # Hidden size for each LSTM layer
num_layers : int = 2       # Number of LSTM layers
dropout : float = 0.2      # Dropout for regularization between LSTM layers
### Training params
loss_fn = nn.CrossEntropyLoss()
learning_rate : float = 5e-4
batch_size : int = 20
epochs : int = 10

###load and pull content
file : str = "data.csv"
csvfile = open(file, "r")
    
readout = list(csv.reader(csvfile))
goongagas = []
for i, r in enumerate(readout):
    if i == 0:  # Skip header row
        continue
    if len(r) > 3 and r[3].strip():  # Skip empty messages
        goongagas.append(r[3].lower())
readout = goongagas
goongagas = None

### begin loading and tokenizing data
x = cT.dynamicTokenize(readout,tokDict=voc)

train_dataSet = integerDataset.textDataset(inSize=inSize,outSize=outSize,
                                 tokenizedData=x[0:len(x)//2],
                                 vocSize=vocSize)
test_dataSet = integerDataset.textDataset(inSize=inSize,outSize=outSize,
                                tokenizedData=x[len(x)//2:],
                                vocSize=vocSize)
train_dataloader = DataLoader(train_dataSet, batch_size=batch_size, 
                              shuffle=True,
                              num_workers=4)
test_dataloader = DataLoader(test_dataSet, batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)




###
class NeuralNetwork(nn.Module):
    def __init__(self, vocSize, inSize, outSize, embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.2):
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
        # x shape: (batch_size, inSize)
        x = self.embedding(x)  # (batch_size, inSize, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, inSize, hidden_size)
        # Use the last timestep output
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        x = self.layerNorm(x)
        logits = self.linear(x)  # (batch_size, outSize * vocSize)
        logits = logits.view(-1, self.outSize, self.vocSize)  # (batch_size, outSize, vocSize)
        return logits




model = NeuralNetwork(vocSize=vocSize, inSize=inSize, outSize=outSize, 
                      embedding_dim=embedding_dim, hidden_size=hidden_size, 
                      num_layers=num_layers, dropout=dropout).to(device)
try:
    print('loading last save')
    model.load_state_dict(torch.load('model.pth'))
except:
    print('loading failed, starting from scratch')
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loopdeloop = loops.trainAndTest(train_dataloader,
                                test_dataloader,
                                model,
                                loss_fn,
                                optimizer)

try:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loopdeloop.train_loop()
        loopdeloop.test_loop()
        torch.save(model.state_dict(),'model.pth')
except KeyboardInterrupt:
    print('interrupted')
    print('saving')
    torch.save(model.state_dict(),'model.pth')

print('fun time')


ine = test_dataSet[1][0]
memory = post.IdsToChrs([ine,],voc)[0]

cont = True
while cont:
    tmp = input('>>')
    if tmp == 'EXIT':
        cont = False
        continue
    tmp = tmp.lower()
    memory = memory[len(tmp):] + tmp
    memory = post.inferenceResponse(model,memory,voc)

#for i in range(0,4096):
    
#    a = logitsToId(model(ine.unsqueeze(0).to(device)),timeSteps=outSize,batchSize=1,vocLen=vocSize)
#    ine = torch.cat([ine[1:], a.squeeze().view(1)])
#    print(charDict[a.to('cpu').view(-1)[0].item()],end='') # pyright: ignore[reportArgumentType]



    
#print(IdsToChrs([test_dataSet[1][0],],voc)[0])
#print(IdsToChrs(a,voc))
#print(IdsToChrs([test_dataSet[1][1],],voc)[0])


print("Done!")
