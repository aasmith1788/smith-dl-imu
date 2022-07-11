#!/bin/bash -l

# TODO: 학습 1회 돌린 이후에 학습시간 나오면 적용할 것
#$ -l h_rt=2:00:00
#$ -N EXPRADG
#$ -o ../../result_qsub/EXP_DG_ang/try12th
#$ -j y    
#$ -m ea
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -pe omp 8
# 메모리에 관해서는 아래 주소를 참고
# https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/#MEMORY

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda/4.9.2
conda activate imu
python EXP_torch_DG_regression_angle.py
