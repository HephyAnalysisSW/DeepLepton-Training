
from DeepJetCore.training.training_base import training_base
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
import tensorflow as tf

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()

layout = {}

layout["activation"] = "elu"

# Pfcands + SV
layout["neutral"]  = [5, 5]
layout["charged"]  = [5, 5]
layout["photons"]  = [5, 5]
layout["elctrons"] = [5, 5]
layout["muons"]    = [5, 5]
layout["SV"]       = [5, 5]
layout["mode"]     = "Dense" # "CNN" # Type of Layer used for pfCands, 
layout["cands"]    = [200, 100] # None # Layout of Dense Layers after LSTMs before concatenation with lepton variables, leave Non if not required

# Lepton Features
layout["leptonStart"] = [100, 100] # None if not wished  

layout["fC"] = [300, 300, 100, 100, 100, 100]



if newtraining:
    from models import buildModel
    
    train.setModel(buildModel, layout=layout, dropoutRate=0.5, momentum=0.2)
    
    
    train.compileModel(learningrate=0.001, #0.001,
                       loss=['categorical_crossentropy'],
                       metrics=[tf.keras.metrics.Accuracy()],
                       )


    train.train_data.maxFilesOpen=25 #5
    
    print(train.keras_model.summary())

    model,history = train.trainModel(nepochs=20, # Test what is best,   
                                     batchsize=10000,
                                     stop_patience=300,
                                     lr_factor=0.5,
                                     lr_patience=-1,
                                     lr_epsilon=0.0001,
                                     lr_cooldown=10,
                                     lr_minimum=0.00001,
                                     verbose=1,checkperiod=1)
