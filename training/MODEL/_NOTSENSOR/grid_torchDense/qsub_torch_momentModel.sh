#!/bin/bash -l

# 1시간 30분짜리(넉넉하게 2시간) 학습 3개 
#$ -l h_rt=5:00:00
#$ -N Dense_M
#$ -o ../../../result_qsub/moment/NOTSENSOR
#$ -j y    
#$ -m ea
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 8
# 메모리에 관해서는 아래 주소를 참고
# https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/#MEMORY

module load miniconda/4.9.2
conda activate imu
python grid_torch_momentModel.py
