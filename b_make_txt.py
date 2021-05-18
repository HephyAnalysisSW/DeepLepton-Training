import os
import random
import argparse


argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--path',
                        action='store',
                        required=True,
                        help='path to files')

argParser.add_argument('--N',
                        action='store',
                        default =0,
                        help='Number of files to write into filelist for training')

argParser.add_argument('--valfrac',
                     action='store',
                     required=True,
                     help='Fraction of files for prediction')

argParser.add_argument('--testfrac',
                     action='store',
                     required=True,
                     help='Fraction of files for final testing')
args = argParser.parse_args()


# Number of files to train on, if N = 0 all files will be used
# if more files are specified than how many exist, all files will be taken
N = int(args.N)

# Split for validation
val = float(args.valfrac)

# Split for prediction (These files will not be trained nor validated on)
test = float(args.testfrac)

# Test if the inputs are somehow reasonable
assert(test>0 and val>0 and test+val<0.5)

# directory where the .root files live
#path = /scratch-cbe/users/benjamin.wilhelmy/DeepLepton/v2/step2/2016/muo/pt_3.5_-1/STopvsTop 
path = args.path

# read .root filenames from directory
filenames = os.listdir(path)
filenames = [f for f in filenames if '.root' in f] 

# Cut filenames to length
if len(filenames) < N:
    print('Not enough files in directory, will use the ones that are\
             available.')
    N_files = len(filenames)
elif N == 0:
    N_files = len(filenames)
else:
    N_files   = int(N)
    random.shuffle(filenames)
    filenames = filenames[:N_files]

# Split into train and predict files
# Checked the seed, ok
random.seed(7)
random.shuffle(filenames)

# Now spilt the files according to their category
n_val = int(N_files*val)
n_test = int(N_files*test)
n_train = N_files - n_val - n_test

print("Found {} files for validation, {} for testing and {} for training".format(n_val,
                                                                    n_test,
                                                                    n_train))

print("The real fraction is {} for n_val and {} for n_test".format(float(n_val)/N_files,
                                                                float(n_test)/N_files))

filenames_train = filenames[:n_train]
filenames_val = filenames[n_train:n_train+n_val ]
filenames_test = filenames[n_train+n_val:]
print("train = {} files, val = {} files, test = {} files".format(len(filenames_train),
         len(filenames_val),
         len(filenames_test)))

# print(filenames_test)
# print(filenames_val)
# print(filenames_train)

categories = ["train", "val", "test"]
outfilename = "files_three_cat_"
for cat in categories:
    if os.path.exists(os.path.join(path, outfilename + cat + '.txt')):
        print('removing old ', os.path.join(path, outfilename + cat + '.txt'))
        os.remove(os.path.join(path, outfilename + cat + '.txt'))
    # print('\n'.join(eval("filenames_{}".format(cat))))
    with open(os.path.join(path, outfilename + cat + '.txt'), 'w') as f:
        # print("filenames_" + cat)
        # filenames_tmp = eval("filenames_{}".format(cat))
        # print(filenames_tmp)
        f.write('\n'.join(eval("filenames_{}".format(cat))))

    print("{} filelist written to:".format(cat), os.path.join(path, outfilename +
            cat + '.txt')) 

