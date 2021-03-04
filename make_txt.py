import os
import random
random.seed(7)

import argparse

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--path', action='store', required=True)
argParser.add_argument('--N',    action='store', default =0   , help='Number of files to write into filelist for training')
argParser.add_argument('--n',    action='store', required=True, help='Proportion of number of files for prediction')
args = argParser.parse_args()


# Number of files to train on, if N = 0 all files will be used
# if more files are specified than how many exist, all files will be taken
N = int(args.N)

# Split for prediction (These files will not be trained or validated on)
n = float(args.n)

# directory where the .root files live
#path = '/eos/vbc/user/maximilian.moser/DeepLepton/v1/step2/2016/muo/pt_3.5_-1/DYvsQCD/'
path = args.path

# read .root filenames from directory
filenames = os.listdir(path)
filenames = [f for f in filenames if '.root' in f] 

# Cut filenames to length
if len(filenames) < N + N*n :
    print('Not enough files in directory, will use the ones that are available.')
    N_files = len(filenames)
elif N == 0:
    N_files = len(filenames)
else:
    N_files   = int(N + N*n)
    random.shuffle(filenames)
    filenames = filenames[:N_files]

# Split into train and predict files
random.shuffle(filenames)
filenames_train   = filenames[ :int( N_files / (1+n) ) ]
filenames_predict = filenames[ int( N_files / (1+n) ): ]


if os.path.exists(os.path.join(path, 'files_train.txt')):
    print('removing old ', os.path.join(path, 'files_train.txt'))
    os.remove(os.path.join(path, 'files_train.txt'))
if os.path.exists(os.path.join(path, 'files_predict.txt')):
    print('removing old ', os.path.join(path, 'files_predict.txt'))
    os.remove(os.path.join(path, 'files_predict.txt'))

with open(os.path.join(path, 'files_train.txt'), 'w') as f:
    f.write('\n'.join(filenames_train))

with open(os.path.join(path, 'files_predict.txt'), 'w') as f:
    f.write('\n'.join(filenames_predict))

print('Train   filelist written to:', os.path.join(path, 'files_train.txt'))
print('Predict filelist written to:', os.path.join(path, 'files_predict.txt'))


