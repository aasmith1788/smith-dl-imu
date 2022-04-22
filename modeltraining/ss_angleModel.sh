#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -N angle
#$ -o result_qsub/angle
#$ -j y    
#$ -m ea
#$ -l gpus=1
#$ -pe omp 8
#$ -l mem_per_core=8G

module load miniconda/4.9.2
conda activate sccIMU

# NSLOTS will be set to the # of requested cores

export TF_NUM_INTEROP_THREADS=$(( $NSLOTS - 1 ))
export TF_NUM_INTRAOP_THREADS=1
python batch_5fold_angleModel.py
