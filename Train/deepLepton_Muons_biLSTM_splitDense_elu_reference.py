import tensorflow as tf

from DeepJetCore.training.training_base import training_base
#from Losses import loss_NLL, loss_meansquared
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepLeptonReference_biLSTM_split_elu
    
    train.setModel(model_deepLeptonReference_biLSTM_split_elu,dropoutRate=0.5,momentum=0.2)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001, #0.001,
                       loss=['categorical_crossentropy'],
                       #loss=['categorical_crossentropy',loss_meansquared],
                       metrics=['accuracy'],
                       loss_weights=[1.]
                       #loss_weights=[1., 0.000000000001]
                       )


    train.train_data.maxFilesOpen=25 #5
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=100, #3, #4 
                                     batchsize=2048, #10000, #64, #512, #1024, #2048, #4096
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=5, 
                                     lr_epsilon=0.00001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.00001, 
                                     maxqsize=25
                                     )
