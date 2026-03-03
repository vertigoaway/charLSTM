import charLSTM as clstm
import trinketbox.ai.utils.outProcessing as post

print('starting terminal interface')
post.basicInterface(clstm.model,clstm.vocab,timeSteps=clstm.inSize)