import charLSTM as clstm
import trinketbox.ai.utils.outProcessing as post

#TODO: use command line args for choosing what weights to load
if __name__=='__main__':
    model = clstm.create()
    model.loadWeights(clstm.modelPath)
    print('starting terminal interface')
    post.basicInterface(model,clstm.vocab,timeSteps=clstm.inSize)