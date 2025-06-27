# Running Neural Network Jobs on the BU SCC

This document collects the practical steps required to execute the training scripts on the Boston University Shared Computing Cluster (SCC). All job scripts in `training/` use the Sun Grid Engine scheduler via `qsub` and assume a Conda environment with PyTorch, TensorFlow and supporting packages.

## 1. Prepare the Conda Environment

Create an environment named `imu` using the YAML file included with the preprocessing notebooks:

```bash
conda env create -n imu -f preperation/buIMU.yml
conda activate imu
```

The environment installs Python 3.8 along with `torch`, `tensorflow`, `numpy`, `pandas` and other dependencies needed for both data preparation and model training.

## 2. Dataset Location

The training scripts expect the processed `.npz` archives and scaler pickles under a shared directory. Paths are configured near the top of each Python file. For example, `training/MODEL/torchDense/torch_angleModel.py` defines:

```python
absDataDir = r'/restricted/project/movelab/bcha/IMUforKnee/preperation/SAVE_dataSet'
SaveDir   = r'/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel'
logDir    = r'/restricted/project/movelab/bcha/IMUforKnee/training/logs'
```

Copy your dataset to the SCC and adjust these variables if your directories differ. Each dataset folder contains files such as `0_fold_final_train.npz` and matching `*_scaler4Y_angle.pkl` pickles.

## 3. Submitting Jobs

Batch scripts under `training/MODEL/**/` configure wall time, GPU resources and output paths. A minimal example is `training/MODEL/torchDense/ss_torch_angleModel.sh`:

```bash
#$ -l h_rt=2:00:00
#$ -N torch_angle
#$ -o ../../result_qsub/angle/exp_5
#$ -j y
#$ -l gpus=1
#$ -pe omp 8
module load miniconda/4.9.2
conda activate imu
python torch_angleModel.py
```

Submit this script with:

```bash
qsub ss_torch_angleModel.sh
```

Log files will appear in the `result_qsub` directory specified by the `-o` option. The Python program writes TensorBoard events to `logDir` and saves TorchScript checkpoints to `SaveDir`.

## 4. Monitoring Training

While jobs run you can inspect progress by viewing the standard output log or by pointing TensorBoard at the log directory:

```bash
tensorboard --logdir /restricted/project/movelab/bcha/IMUforKnee/training/logs
```

Each fold logs loss and per-axis normalized RMSE. Hyperparameters such as learning rate and batch size are recorded via TensorBoard’s HParams API.

## 5. Customizing Experiments

- Edit `nameDataset`, `learningRate`, `batch_size` and other variables inside the Python scripts before submitting jobs.
- Modify the `#$ -l h_rt` and `#$ -pe omp` directives in the `.sh` files to request different runtimes or CPU thread counts.
- Update the `-o` path to keep results from separate experiments organized.

## 6. After Training

Trained models are saved under the directory specified by `SaveDir` using file names like `angle_0_fold.pt`. Estimation notebooks in `estimation/` load these checkpoints to generate predictions and compute summary tables. Ensure the notebooks reference the same dataset name and path as the training scripts.

---
This guide consolidates the cluster-related details spread throughout the repository so new users can reproduce the neural network experiments on the SCC.
