#!/bin/bash -l

# 보통 torch_VAE_LSTM.py는 1개 돌리는데 1시간 정도 걸리는듯함.
#$ -l h_rt=15:00:00
#$ -N VL
#$ -o ../../../result_qsub/vaelstm/NOTSENSOR
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
python torch_VAE_LSTM.py
