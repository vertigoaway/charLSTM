import trinketbox.ai.utils.NNLoops as loops
import charLSTM as clstm
from torch import nn
import torch

model = clstm.model


batch_size : int = 20
epochs : int = 10
trainingDataPath : str = "data.csv"

lossFn = nn.CrossEntropyLoss()
learning_rate : float = 5e-4

optimizer = torch.optim.Adam(clstm.model.parameters(), lr=learning_rate)



train_dataloader,test_dataloader = clstm.loadTrainAndTestData(trainingDataPath=trainingDataPath,batch_size=batch_size)
loopdeloop = loops.trainAndTest(train_dataloader,
                                test_dataloader,
                                clstm.model,
                                lossFn,
                                optimizer)

print('starting training session')
print(f"Approx training steps per epoch:{len(train_dataloader)//batch_size}")
try:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loopdeloop.train_loop()
        loopdeloop.test_loop()
        print('saving model')
        torch.save(model.state_dict(),clstm.modelPath)
except KeyboardInterrupt:
    print('interrupted!')
    print('saving model')
    torch.save(model.state_dict(),clstm.modelPath)
print('training session finished')
