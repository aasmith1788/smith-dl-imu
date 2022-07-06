#!/bin/bash -l

#* 이 코드는 angle에서 성공했을 때 해당 시도를 moment에 적용하는 것을 목적으로 한다
#! 절대로 검토없이 돌리지 말것
#? angle은 성공할까
# TODO 검토하기
#$ -l h_rt=4:00:00
#$ -N RMDG
#$ -o ../../result_qsub/EXP_DG_mo/
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
python EXP_torch_DG_regression_moBWHT.py
