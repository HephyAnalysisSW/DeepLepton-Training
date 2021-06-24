from __future__ import print_function
from DeepJetCore.training.training_base import training_base

import numpy as np
import tensorflow as tf

from models import *

from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM, Convolution1D, Conv2D, Bidirectional, MaxPooling1D, Activation, Lambda
from keras.models import Model
from keras.layers import BatchNormalization

def makeDense(neutral, charged, photon, electron, muon, vertices, layout, dropoutRate=0.5, batchnorm=True, batchmomentum=0.2):
    print('making dense')    
    npf = neutral
    cpf = charged
    ppf = photon
    epf = electron
    mpf = muon
    vtx = vertices

    npf_shape = layout["neutral"]
    cpf_shape = layout["charged"]
    ppf_shape = layout["photon"]
    epf_shape = layout["electron"]
    mpf_shape = layout["muon"]
    vtx_shape = layout["SV"]

    activation = layout["activation"]

    ctr = 0
    for N in npf_shape:
        npf = Dense(N, kernel_initializer='lecun_uniform',  activation=activation, name='npf_dense'+str(ctr))(npf)
        if batchnorm:
            if ctr < len(npf_shape)-1: # last Dense layer should not be batch normed
                npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm'+str(ctr))(npf)
        ctr += 1

    ctr = 0
    for N in cpf_shape:
        cpf = Dense(N, kernel_initializer='lecun_uniform',  activation=activation, name='cpf_dense'+str(ctr))(cpf)
        if batchnorm:
            if ctr < len(cpf_shape)-1: # last Dense layer should not be batch normed
                cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm'+str(ctr))(cpf)
        ctr += 1

    ctr = 0
    for N in ppf_shape:
        ppf = Dense(N, kernel_initializer='lecun_uniform',  activation=activation, name='ppf_dense'+str(ctr))(ppf)
        if batchnorm:
            if ctr < len(ppf_shape)-1: # last Dense layer should not be batch normed
                ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm'+str(ctr))(ppf)
        ctr += 1

    ctr = 0
    for N in epf_shape:
        epf = Dense(N, kernel_initializer='lecun_uniform',  activation=activation, name='epf_dense'+str(ctr))(epf)
        if batchnorm:
            if ctr < len(epf_shape)-1: # last Dense layer should not be batch normed
                epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm'+str(ctr))(epf)
        ctr += 1

    ctr = 0
    for N in mpf_shape:
        mpf = Dense(N, kernel_initializer='lecun_uniform',  activation=activation, name='mpf_dense'+str(ctr))(mpf)
        if batchnorm:
            if ctr < len(mpf_shape)-1: # last Dense layer should not be batch normed
                mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm'+str(ctr))(mpf)
        ctr += 1

    ctr = 0
    for N in vtx_shape:
        vtx = Dense(N, kernel_initializer='lecun_uniform',  activation=activation, name='vtx_dense'+str(ctr))(vtx)
        if batchnorm:
            if ctr < len(vtx_shape)-1: # last Dense layer should not be batch normed
                vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm'+str(ctr))(vtx)
        ctr += 1


    return npf, cpf, ppf, epf, mpf, vtx

def makeCNN(neutral, charged, photon, electron, muon, vertices, layout, dropoutRate=0.5, batchnorm=True, batchmomentum=0.2):
    print('making dense')    
    npf = neutral
    cpf = charged
    ppf = photon
    epf = electron
    mpf = muon
    vtx = vertices

    npf_shape = layout["neutral"]
    cpf_shape = layout["charged"]
    ppf_shape = layout["photon"]
    epf_shape = layout["electron"]
    mpf_shape = layout["muon"]
    vtx_shape = layout["SV"]

    activation = layout["activation"]

    ctr = 0
    for N in npf_shape:
        npf = Convolution1D(N, 1, kernel_initializer='lecun_uniform',  activation=activation, name='npf_dense'+str(ctr))(npf)
        if batchnorm:
            if ctr < len(npf_shape)-1: # last Convolution1D layer should not be batch normed
                npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm'+str(ctr))(npf)
        ctr += 1

    ctr = 0
    for N in cpf_shape:
        cpf = Convolution1D(N, 1, kernel_initializer='lecun_uniform',  activation=activation, name='cpf_dense'+str(ctr))(cpf)
        if batchnorm:
            if ctr < len(cpf_shape)-1: # last Convolution1D layer should not be batch normed
                cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm'+str(ctr))(cpf)
        ctr += 1

    ctr = 0
    for N in ppf_shape:
        ppf = Convolution1D(N, 1, kernel_initializer='lecun_uniform',  activation=activation, name='ppf_dense'+str(ctr))(ppf)
        if batchnorm:
            if ctr < len(ppf_shape)-1: # last Convolution1D layer should not be batch normed
                ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm'+str(ctr))(ppf)
        ctr += 1

    ctr = 0
    for N in epf_shape:
        epf = Convolution1D(N, 1, kernel_initializer='lecun_uniform',  activation=activation, name='epf_dense'+str(ctr))(epf)
        if batchnorm:
            if ctr < len(epf_shape)-1: # last Convolution1D layer should not be batch normed
                epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm'+str(ctr))(epf)
        ctr += 1

    ctr = 0
    for N in mpf_shape:
        mpf = Convolution1D(N, 1, kernel_initializer='lecun_uniform',  activation=activation, name='mpf_dense'+str(ctr))(mpf)
        if batchnorm:
            if ctr < len(mpf_shape)-1: # last Convolution1D layer should not be batch normed
                mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm'+str(ctr))(mpf)
        ctr += 1

    ctr = 0
    for N in vtx_shape:
        vtx = Convolution1D(N, 1, kernel_initializer='lecun_uniform',  activation=activation, name='vtx_dense'+str(ctr))(vtx)
        if batchnorm:
            if ctr < len(vtx_shape)-1: # last Convolution1D layer should not be batch normed
                vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm'+str(ctr))(vtx)
        ctr += 1


    return npf, cpf, ppf, epf, mpf, vtx




def buildModel(Inputs, layout=None, dropoutRate=0.5, momentum=0.2):
    """
    This will build a Network with the approximate layout of Tim Bruecklers DA figure 4.5 on page 45
    """
    if layout == None:
        raise NotImplementedError

    NoCands = False # use PfCandidates if False
    try:
        if layout["NoCands"]:
            NoCands = True
        else:
            NoCands = False
    except:
        pass
    
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0]) # Lepton vars
    if not NoCands:
        print("using Cands")
        npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1]) # neutral
        cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2]) # charged
        ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3]) # photon
        epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4]) # electron
        mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5]) # muon
        vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6]) # SV
    
        if layout["mode"] == "Dense":
            npf, cpf, ppf, epf, mpf, vtx = makeDense(neutral=npf, 
                                                 charged=cpf, 
                                                 photon=ppf, 
                                                 electron=epf, 
                                                 muon=mpf, 
                                                 vertices=vtx, 
                                                 layout=layout, 
                                                 dropoutRate=dropoutRate, 
                                                 batchnorm=True, 
                                                 batchmomentum=momentum)
        elif layout["mode"] == "CNN":
            npf, cpf, ppf, epf, mpf, vtx = makeCNN(neutral=npf, 
                                                 charged=cpf, 
                                                 photon=ppf, 
                                                 electron=epf, 
                                                 muon=mpf, 
                                                 vertices=vtx, 
                                                 layout=layout, 
                                                 dropoutRate=dropoutRate, 
                                                 batchnorm=True, 
                                                 batchmomentum=momentum)
        else:
            raise NotImplementedError("Implement this Layer type!")

        # LSTMs:
        npf = Bidirectional(LSTM(50,implementation=2, name='npf_lstm'), merge_mode='concat')(npf)
        npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
        npf = Dropout(dropoutRate)(npf)
                
        cpf = Bidirectional(LSTM(150,implementation=2, name='cpf_lstm'), merge_mode='concat')(cpf)
        cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
                                
        ppf = Bidirectional(LSTM(50,implementation=2, name='ppf_lstm'), merge_mode='concat')(ppf)
        ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
        ppf = Dropout(dropoutRate)(ppf)
                                                    
        epf = Bidirectional(LSTM(50,implementation=2, name='epf_lstm'), merge_mode='concat')(epf)
        epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
        epf = Dropout(dropoutRate)(epf)
                                                                
        mpf = Bidirectional(LSTM(50,implementation=2, name='mpf_lstm'), merge_mode='concat')(mpf)
        mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
        mpf = Dropout(dropoutRate)(mpf)
                                                                                
        vtx = Bidirectional(LSTM(150,implementation=2, name='vtx_lstm'), merge_mode='concat')(vtx)
        vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
        vtx = Dropout(dropoutRate)(vtx)

    
        # Join pfCands and SV
        xCands   = Concatenate()( [npf, cpf, ppf, epf, mpf, vtx] )
    
        if layout["cands"] != None:
            ctr = 0
            for N in layout["cands"]:
                xCands = Dense(N, activation=layout["activation"], kernel_initializer="lecun_uniform", name="cands_Dense"+str(ctr))(xCands)
                xCands = BatchNormalization(momentum=momentum,name="cands_dense_batchnorm"+str(ctr))(xCands)
                xCands = Dropout(dropoutRate, name="cands_dense_dropout"+str(ctr))(xCands)
                ctr += 1
    
    xGlobals = globalvars
    if layout["leptonStart"] != None:
        ctr = 0
        for N in layout["leptonStart"]:
            xGlobals = Dense(N, activation=layout["activation"], kernel_initializer="lecun_uniform", name="global_Dense"+str(ctr))(xGlobals)
            xGlobals = BatchNormalization(momentum=momentum,name="global_dense_batchnorm"+str(ctr))(xGlobals)
            xGlobals = Dropout(dropoutRate, name="global_dense_dropout"+str(ctr))(xGlobals)
            ctr += 1

    # Join Cands and Global(lepton vars):
    if not NoCands:
        x = Concatenate()( [xGlobals,xCands] )
    else:
        x = xGlobals

    # Last part of Dense Layers
    
    ctr = 0
    for N in layout["fC"]:
        x = Dense(N, activation=layout["activation"], kernel_initializer="lecun_uniform", name="Dense"+str(ctr))(x)
        x = BatchNormalization(momentum=momentum,name="dense_batchnorm"+str(ctr))(x)
        x = Dropout(dropoutRate, name="dense_dropout"+str(ctr))(x)
        ctr += 1
    # Prediction layer
    lepton_pred=Dense(3, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [lepton_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    
    return model


