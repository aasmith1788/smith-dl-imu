#!/bin/bash -l

#$ -l h_rt=00:10:00
#$ -N tmp
#$ -o tmp
#$ -j y    
#$ -m ea
#$ -l gpus=1
#$ -pe omp 4
# 메모리에 관해서는 아래 주소를 참고
# https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/#MEMORY


module load miniconda/4.9.2
conda activate torchIMU

# NSLOTS will be set to the # of requested cores

export TF_NUM_INTEROP_THREADS=$(( $NSLOTS - 1 ))
export TF_NUM_INTRAOP_THREADS=1
python tmp_torch.py
