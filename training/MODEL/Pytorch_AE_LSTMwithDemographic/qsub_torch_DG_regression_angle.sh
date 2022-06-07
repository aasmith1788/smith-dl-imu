#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -N regAng
#$ -o ../../result_qsub/regAng/DG_try_01
#$ -j y    
#$ -m ea
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 8
# 메모리에 관해서는 아래 주소를 참고
# https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/#MEMORY

module load miniconda/4.9.2
conda activate imu
python torch_DG_regression_angle.py
