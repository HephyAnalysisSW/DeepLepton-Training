#! /bin/sh

echo "starting environment"

source /users/maximilian.moser/DeepLeptonStuff/DeepLepton-Training/env.sh

echo "starting conversion"

convertFromSource.py -i /scratch-cbe/users/maximilian.moser/DeepLepton/traindata/DYvsQCD_2016/files_train.txt -o /scratch-cbe/users/maximilian.moser/DeepLepton/Train_DYvsQCD_2016/ -c TrainDataDeepLepton
