#! /bin/sh -x
#SBATCH --job-name=convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --qos=medium

SIF='/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest'


data_dir="/scratch-cbe/users/${USER}/DeepLepton"

input_data="${data_dir}/traindata/DYvsQCD_2016/files_train.txt"
output_dir="${data_dir}/Train_DYvsQCD_rH_flat/"


if [ ! -e $input_data ]; then
    echo "Input data not found."
    exit 1
fi

if [ -e $output_dir ]; then
    echo "Output directory already exists"
    exit 1
fi

nvidia-smi

singularity run $SIF <<EOF
set -x
source env.sh
convertFromSource.py -i $input_data -o $output_dir -c TrainDataDeepLepton --noramcopy
EOF
