#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -N angle
#$ -o ../../result_qsub/angle_lean
#$ -j y    
#$ -m ea
#$ -l gpus=1
#$ -pe omp 4

module load miniconda/4.9.2
conda activate scc

# NSLOTS will be set to the # of requested cores

export TF_NUM_INTEROP_THREADS=$(( $NSLOTS - 1 ))
export TF_NUM_INTRAOP_THREADS=1
python angle_Model.py
