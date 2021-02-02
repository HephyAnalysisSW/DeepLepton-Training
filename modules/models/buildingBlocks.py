'''
standardised building blocks for the models
'''
from DeepJetCore.training.training_base import training_base

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D
#from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers import BatchNormalization

##Deep Lepton Standard Architecture
#def block_deepLeptonConvolutions(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    '''
#    deep Lepton convolution part. 
#    '''
#    npf=neutrals
#    if active:
#        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
#        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
#        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
#    else:
#        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
#    
#    cpf=charged
#    if active:
#        cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
#        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
#        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
#        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
#    else:
#        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
#        
#    ppf=photons
#    if active:
#        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv0')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
#        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv1')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
#        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='ppf_conv2')(ppf)
#    else:
#        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)
#
#    epf=electrons
#    if active:
#        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv0')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
#        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv1')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
#        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='epf_conv2')(epf)
#    else:
#        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)
#
#    mpf=muons
#    if active:
#        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv0')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
#        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv1')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
#        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='mpf_conv2')(mpf)
#    else:
#        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)
#
#    vtx = vertices
#    if active:
#        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)
#
#    return npf,cpf,ppf,epf,mpf,vtx
#
#def block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
##Deep Lepton Standard Architecture with selu activation function
#def block_deepLeptonConvolutions_selu(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    '''
#    deep Lepton convolution part. 
#    '''
#    npf=neutrals
#    if active:
#        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='selu', name='npf_conv0')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
#        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='selu', name='npf_conv1')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
#        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='selu' , name='npf_conv2')(npf)
#    else:
#        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
#    
#    cpf=charged
#    if active:
#        cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='selu', name='cpf_conv0')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
#        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='selu', name='cpf_conv1')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
#        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='selu', name='cpf_conv2')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
#        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='selu' , name='cpf_conv3')(cpf)
#    else:
#        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
#        
#    ppf=photons
#    if active:
#        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='selu', name='ppf_conv0')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
#        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='selu', name='ppf_conv1')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
#        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='selu' , name='ppf_conv2')(ppf)
#    else:
#        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)
#
#    epf=electrons
#    if active:
#        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='selu', name='epf_conv0')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
#        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='selu', name='epf_conv1')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
#        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='selu' , name='epf_conv2')(epf)
#    else:
#        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)
#
#    mpf=muons
#    if active:
#        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='selu', name='mpf_conv0')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
#        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='selu', name='mpf_conv1')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
#        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='selu' , name='mpf_conv2')(mpf)
#    else:
#        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)
#
#    vtx = vertices
#    if active:
#        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='selu', name='vtx_conv0')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='selu', name='vtx_conv1')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='selu', name='vtx_conv2')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='selu', name='vtx_conv3')(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)
#
#    return npf,cpf,ppf,epf,mpf,vtx
#
#def block_deepLeptonDense_selu(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='selu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
#Deep Lepton Standard Architecture with elu activation function
def block_deepLeptonConvolutions_elu(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
    '''
    deep Lepton convolution part. 
    '''
    npf=neutrals
    if active:
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
    
    cpf=charged
    if active:
        cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='elu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    ppf=photons
    if active:
        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='ppf_conv0')(ppf)
        if batchnorm:
            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='ppf_conv1')(ppf)
        if batchnorm:
            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='ppf_conv2')(ppf)
    else:
        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)

    epf=electrons
    if active:
        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='epf_conv0')(epf)
        if batchnorm:
            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='epf_conv1')(epf)
        if batchnorm:
            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='epf_conv2')(epf)
    else:
        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)

    mpf=muons
    if active:
        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='mpf_conv0')(mpf)
        if batchnorm:
            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='mpf_conv1')(mpf)
        if batchnorm:
            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='mpf_conv2')(mpf)
    else:
        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)

    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return npf,cpf,ppf,epf,mpf,vtx

def block_deepLeptonDense_elu(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
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
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

##Test max pooling layers
#def block_deepLeptonConvolutions_pooling(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    '''
#    deep Lepton convolution part. 
#    '''
#    npf=neutrals
#    if active:
#        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
#        npf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='npf_pool0')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
#        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
#        npf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='npf_pool1')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
#        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
#        npf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='npf_pool2')(npf)
#    else:
#        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
#    
#    cpf=charged
#    if active:
#        cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
#        cpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='cpf_pool0')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
#        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
#        cpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='cpf_pool1')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
#        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
#        cpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='cpf_pool2')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
#        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
#        cpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='cpf_pool3')(cpf)
#    else:
#        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
#        
#    ppf=photons
#    if active:
#        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv0')(ppf)
#        ppf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='ppf_pool0')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
#        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv1')(ppf)
#        ppf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='ppf_pool1')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
#        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='ppf_conv2')(ppf)
#        ppf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='ppf_pool2')(ppf)
#    else:
#        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)
#
#    epf=electrons
#    if active:
#        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv0')(epf)
#        epf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='epf_pool0')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
#        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv1')(epf)
#        epf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='epf_pool1')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
#        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='epf_conv2')(epf)
#        epf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='epf_pool2')(epf)
#    else:
#        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)
#
#    mpf=muons
#    if active:
#        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv0')(mpf)
#        mpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='mpf_pool0')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
#        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv1')(mpf)
#        mpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='mpf_pool1')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
#        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='mpf_conv2')(mpf)
#        mpf = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='mpf_pool2')(mpf)
#    else:
#        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)
#
#    vtx = vertices
#    if active:
#        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
#        vtx = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='vtx_pool0')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
#        vtx = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='vtx_pool1')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
#        vtx = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='vtx_pool2')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
#        vtx = MaxPooling1D(pool_length=2, stride=None, border_mode='same', name='vtx_pool3')(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)
#
#    return npf,cpf,ppf,epf,mpf,vtx
#
#
##Deep Lepton Test Architecture
#def block_deepLeptonConvolutions_testSize(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    '''
#    deep Lepton convolution part. 
#    '''
#    npf=neutrals
#    if active:
#        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
#        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
#        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
#    else:
#        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
#    
#    cpf=charged
#    if active:
#        cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
#        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
#        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
#        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
#    else:
#        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
#        
#    ppf=photons
#    if active:
#        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv0')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
#        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv1')(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
#        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='ppf_conv2')(ppf)
#    else:
#        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)
#
#    epf=electrons
#    if active:
#        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv0')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
#        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv1')(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
#        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='epf_conv2')(epf)
#    else:
#        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)
#
#    mpf=muons
#    if active:
#        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv0')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
#        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv1')(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
#        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='mpf_conv2')(mpf)
#    else:
#        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)
#
#    vtx = vertices
#    if active:
#        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)
#
#    return npf,cpf,ppf,epf,mpf,vtx
#
#def block_deepLeptonDense_testSize(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(400, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
#
##DeepLepton Test Architecture for Split DNN for globalVars and pfCands+SV and concatenate both networks in final dense layers
#def block_deepLeptonDense_testSplit_sum(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
#def block_deepLeptonDense_testSplit_cands(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_cands_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout0')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_cands_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout1')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_cands_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout2')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_cands_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_cands_dense_off')(x)
#    
#    return x
#
#def block_deepLeptonDense_testSplit_global(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_global_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout0')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_global_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout1')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_global_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout2')(x)
#        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_global_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_global_dense_off')(x)
#    
#    return x
#
##DeepLepton Test Architecture for Split DNN for globalVars and pfCands+SV and concatenate both networks in final dense layers + selu activation function
#def block_deepLeptonDense_testSplit_selu_sum(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='selu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
#def block_deepLeptonDense_testSplit_selu_cands(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='selu',kernel_initializer='lecun_uniform', name='df_cands_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout0')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_cands_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout1')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_cands_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout2')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_cands_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_cands_dense_off')(x)
#    
#    return x
#
#def block_deepLeptonDense_testSplit_selu_global(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='selu',kernel_initializer='lecun_uniform', name='df_global_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout0')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_global_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout1')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_global_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout2')(x)
#        x=  Dense(100, activation='selu',kernel_initializer='lecun_uniform', name='df_global_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_global_dense_off')(x)
#    
#    return x
#
##TimTest deepLepton Blocks
#import keras.regularizers
#def block_deepLeptonConvolutions_Tim(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    '''
#    deep Lepton convolution part. 
#    '''
#    npf=neutrals
#    if active:
#        npf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
#        npf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
#        npf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(npf)
#    else:
#        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
#    
#    cpf=charged
#    if active:
#        cpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
#        cpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
#        cpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
#        cpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(cpf)
#    else:
#        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
#        
#    ppf=photons
#    if active:
#        ppf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
#        ppf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
#        ppf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='ppf_conv2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(ppf)
#    else:
#        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)
#
#    epf=electrons
#    if active:
#        epf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
#        epf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
#        epf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='epf_conv2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(epf)
#    else:
#        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)
#
#    mpf=muons
#    if active:
#        mpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
#        mpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
#        mpf = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='mpf_conv2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(mpf)
#    else:
#        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)
#
#    vtx = vertices
#    if active:
#        vtx = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)
#
#    return npf,cpf,ppf,epf,mpf,vtx
#
#def block_deepLeptonDense_Tim(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(4, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
#        x=  Dense(2, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l1(0.0001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
##DeepLepton Test Architecture for Split DNN for globalVars and pfCands+SV and concatenate both networks in final dense layers + elu activation function
#def block_deepLeptonDense_testSplit_elu_sum(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
#    
#    return x
#
def block_deepLeptonDense_testSplit_elu_cands(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
    if active:
        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_cands_dense_dropout0')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_cands_dense_dropout1')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_cands_dense_dropout2')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_cands_dense_dropout3')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_cands_dense_off')(x)
    
    return x

def block_deepLeptonDense_testSplit_elu_global(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
    if active:
        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_global_dense_dropout0')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_global_dense_dropout1')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_global_dense_dropout2')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_global_dense_dropout3')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_global_dense_off')(x)
    
    return x

#DeepLepton Test Architecture for Split DNN for globalVars and pfCands+SV and concatenate both networks in final dense layers + elu activation function + regularization
def block_deepLeptonDense_testSplit_elu_sum_2017(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
    if active:
        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_dense0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_dense2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        e=  Dense(10, activation='elu',kernel_initializer='lecun_uniform', name='df_dense3', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

#def block_deepLeptonDense_testSplit_elu_cands_2017(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout0')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout1')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout2')(x)
#        x=  Dense(10, activation='elu',kernel_initializer='lecun_uniform', name='df_cands_dense3', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_cands_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_cands_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_cands_dense_off')(x)
#    
#    return x
#
#def block_deepLeptonDense_testSplit_elu_global_2017(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    if active:
#        x=  Dense(200, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm0')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout0')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm1')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout1')(x)
#        x=  Dense(100, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm2')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout2')(x)
#        x=  Dense(10, activation='elu',kernel_initializer='lecun_uniform', name='df_global_dense3', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(x)
#        if batchnorm:
#            x = BatchNormalization(momentum=batchmomentum,name='df_global_dense_batchnorm3')(x)
#        x = Dropout(dropoutRate,name='df_global_dense_dropout3')(x)
#    else:
#        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_global_dense_off')(x)
#    
#    return x
#
#
##Deep Lepton Standard Architecture with elu activation function + regularaization
#def block_deepLeptonConvolutions_2017(charged,neutrals,photons,electrons,muons,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.2):
#    '''
#    deep Lepton convolution part. 
#    '''
#    npf=neutrals
#    if active:
#        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='npf_conv0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
#        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='npf_conv1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(npf)
#        if batchnorm:
#            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
#        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
#        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='npf_conv2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(npf)
#    else:
#        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
#    
#    cpf=charged
#    if active:
#        cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='elu', name='cpf_conv0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
#        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='cpf_conv1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf) 
#        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='cpf_conv2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(cpf)
#        if batchnorm:
#            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
#        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)
#        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='cpf_conv3', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(cpf)
#    else:
#        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
#        
#    ppf=photons
#    if active:
#        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='ppf_conv0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
#        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='ppf_conv1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(ppf)
#        if batchnorm:
#            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
#        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
#        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='ppf_conv2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(ppf)
#    else:
#        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)
#
#    epf=electrons
#    if active:
#        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='epf_conv0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
#        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='epf_conv1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(epf)
#        if batchnorm:
#            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
#        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
#        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='epf_conv2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(epf)
#    else:
#        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)
#
#    mpf=muons
#    if active:
#        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='mpf_conv0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
#        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='mpf_conv1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(mpf)
#        if batchnorm:
#            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
#        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
#        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu' , name='mpf_conv2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(mpf)
#    else:
#        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)
#
#    vtx = vertices
#    if active:
#        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv0', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv1', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv2', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='elu', name='vtx_conv3', bias_regularizer=keras.regularizers.l2(0.00000001), kernel_regularizer=keras.regularizers.l2(0.00000001), activity_regularizer=keras.regularizers.l1(0.00000001))(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)
#
#    return npf,cpf,ppf,epf,mpf,vtx
