#!/bin/bash
#SBATCH --job-name=adultnn
#SBATCH --output=adultdatasetgrid.out
#SBATCH --error=adultdatasetgrid.err
#SBATCH -p reservation
##SBATCH --reservation=CSYE7374_GPU
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem=10000
#SBATCH -n 12


module load python/3.6.6
module load cuda/9.0


chdir=$HOME/csye7374-boranchac/FinalProject/
cd $chdir

python adult_nn.py

