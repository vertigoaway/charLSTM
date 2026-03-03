import trinketbox.ai.utils.NNLoops as loops
import charLSTM as clstm

import torch
model = clstm.model
optimizer = clstm.optimizer(clstm.model.parameters(), lr=clstm.learning_rate)

loopdeloop = loops.trainAndTest(clstm.train_dataloader,
                                clstm.test_dataloader,
                                clstm.model,
                                clstm.lossFn,
                                optimizer)

print('starting training session')
print(f"Approx training steps per epoch:{len(clstm.train_dataSet)//clstm.batch_size}")
try:
    for t in range(clstm.epochs):
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
