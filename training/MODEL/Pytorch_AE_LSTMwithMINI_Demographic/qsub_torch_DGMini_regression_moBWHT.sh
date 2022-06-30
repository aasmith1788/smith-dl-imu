#!/bin/bash -l

#$ -l h_rt=2:00:00
#$ -N MNmo
#$ -o ../../result_qsub/miniDgRegMo/20220629_weightDecay_0.0005
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
python torch_DGMini_regression_moBWHT.py
