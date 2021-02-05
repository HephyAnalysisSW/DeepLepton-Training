

from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, Bidirectional, LSTM, Convolution1D, Concatenate

def my_model(Inputs,otheroption):
   

 
    #x = Inputs[0] #this is the self.x list from the TrainData data structure
    #x = Dense(32, activation='relu')(x)

    active = True
    batchnorm=False
    batchmomentum=0.2
    dropoutRate=0.5


    globalvars = BatchNormalization(momentum=batchmomentum,name='globals_input_batchnorm') (Inputs[0])

    npf    =     BatchNormalization(momentum=batchmomentum,name='npf_input_batchnorm')     (Inputs[1])
    if False:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
    print ("1")
    npf = Bidirectional(LSTM(50,implementation=2, name='npf_lstm'), merge_mode='concat')(npf)
    print ("2")
    npf = BatchNormalization(momentum=batchmomentum,name='npflstm_batchnorm')(npf)
    print ("3")
    npf = Dropout(dropoutRate)(npf)
    print ("4")

    xCands  = npf #Concatenate()( [npf] )

#    if False:
#        xCands=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense0')(xCands)
#        if batchnorm:
#            xCands = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm0')(xCands)
#        xCands = Dropout(dropoutRate,name='df_cands_dense_dropout0')(xCands)
#        xCands=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense1')(xCands)
#        if batchnorm:
#            xCands = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm1')(xCands)
#        xCands = Dropout(dropoutRate,name='df_cands_dense_dropout1')(xCands)
#        xCands=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense2')(xCands)
#        if batchnorm:
#            xCands = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm2')(xCands)
#        xCands = Dropout(dropoutRate,name='df_cands_dense_dropout2')(xCands)
#        xCands=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense3')(xCands)
#        if batchnorm:
#            xCands = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm3')(xCands)
#        xCands = Dropout(dropoutRate,name='df_cands_dense_dropout3')(xCands)
#    else:
#        xCands= Dense(1,kernel_initializer='zeros',trainable=False,name='df_cands_dense_off')(xCands)

    if active:
        globalvars=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense0')(globalvars)
        if batchnorm:
            globalvars = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm0')(globalvars)
        globalvars = Dropout(dropoutRate,name='df_global_dense_dropout0')(globalvars)
        globalvars=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense1')(globalvars)
        if batchnorm:
            globalvars = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm1')(globalvars)
        globalvars = Dropout(dropoutRate,name='df_global_dense_dropout1')(globalvars)
        globalvars=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense2')(globalvars)
        if batchnorm:
            globalvars = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm2')(globalvars)
        globalvars = Dropout(dropoutRate,name='df_global_dense_dropout2')(globalvars)
        globalvars=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense3')(globalvars)
        if batchnorm:
            globalvars = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm3')(globalvars)
        globalvars = Dropout(dropoutRate,name='df_global_dense_dropout3')(globalvars)
    else:
        globalvars= Dense(1,kernel_initializer='zeros',trainable=False,name='df_global_dense_off')(globalvars)

    x       = Concatenate()( [globalvars,xCands])

    if active:
        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)

    # 3 prediction classes
    x = Dense(3, activation='softmax')(x)
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)
    
    train.compileModel(learningrate=0.003,
                   loss='categorical_crossentropy') 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=10, 
                                 batchsize=500,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)
                                 
print('Since the training is done, use the predict.py script to predict the model output on your test sample, e.g.: predict.py <training output>/KERAS_model.h5 <training output>/trainsamples.djcdc <path to data>/test.txt <output dir>')
