#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-a10-005&!qa-rtx6k-044&!qa-a10-006
#$ -e errors/
#$ -N safe_100_epoch

# Required modules
module load conda
conda init bash
source activate safe

python train.py