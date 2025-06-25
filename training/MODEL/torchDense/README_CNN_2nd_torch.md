# CNN_2nd_torch Implementation Guide

This document explains how to configure and train the **CNN_2nd_torch** version of `torch_angleModel.py`.  The workflow mirrors the other PyTorch scripts but uses a deeper 1D convolutional network.  It assumes you already generated the dataset under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.

## 1. Environment Setup

1. **Python**: version 3.8 or newer.
2. **Packages**: `torch`, `numpy`, `tqdm`, `tensorboard`.  Install with
   ```bash
   pip install torch numpy tqdm tensorboard
   ```

## 2. Dataset Format

Training relies on five-fold cross validation.  Each fold resides in
`preperation/SAVE_dataSet/<DATASET_NAME>/` and contains `final_X_train`,
`final_X_test`, and target arrays.  Inputs are `(N, 4242, 1)` and targets are `(N, 303, 1)` which reshapes to `(3,101)`【F:training/MODEL/torchDense/README_CNN_1st_torch.md†L25-L36】.
Scaler pickles (`*_fold_scaler4Y_*.pkl`) accompany each fold.

## 3. Training Steps

1. **Prepare files** – Place the `.npz` archives and `.pkl` scalers for all five
   folds under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.
2. **Edit `torch_angleModel.py`** – Set the dataset and choose the model version.
   Lines 16‑39 show the key variables【F:training/MODEL/torchDense/torch_angleModel.py†L16-L39】.
   Change `modelVersion` to `"CNN_2nd_torch"` and adjust paths if training locally.
3. **Launch training**
   - Locally: `python training/MODEL/torchDense/torch_angleModel.py`
   - On BU SCC: `qsub ss_torch_angleModel.sh`
   Each fold is trained sequentially and TensorBoard logs are stored using the
   directory template at lines 144‑149【F:training/MODEL/torchDense/torch_angleModel.py†L144-L149】.

## 4. Model Architecture

When `modelVersion == "CNN_2nd_torch"` the network consists of three
`Conv1d` blocks followed by two dense layers.  The default configuration is:

1. `Conv1d(42, 64, kernel_size=3)` → `ReLU`
2. `Conv1d(64, 64, kernel_size=3)` → `ReLU` → `Dropout(0.3)`
3. `Conv1d(64, 32, kernel_size=3)` → `ReLU`
4. `Flatten`
5. `Linear(32 × 95, 512)` → `ReLU`
6. `Linear(512, 303)`

With no padding each convolution shortens the temporal dimension from 101 to 95
before flattening.  The output vector of length 303 is reshaped to `(3,101)` for
metric evaluation.

## 5. Loss, Optimizer and Training Loop

- **Loss**: custom RMSE as implemented around lines 102‑120 of the script【F:training/MODEL/torchDense/torch_angleModel.py†L102-L120】.
- **Optimizer**: `torch.optim.NAdam`.
- **Hyperparameters**: learning rate 0.0005, batch size 64, epochs 1000 (modifiable in the settings block).

Training and test metrics are logged each epoch.  The routine for saving the
TorchScript model appears at lines 234‑237【F:training/MODEL/torchDense/torch_angleModel.py†L234-L237】.

## 6. Output and Visualization

Run TensorBoard pointing to the log directory to monitor progress:
```bash
tensorboard --logdir <LOG_DIR>
```
Summary writers create subfolders under `<logDir>/<exp_name>/CNN_2nd_torch/...`
for each fold.

## 7. Loading and Evaluating a Trained Model

Use `torch.jit.load` to load the saved `.pt` file, then call `.eval()` before
inference.  The notebook `training/StudyRoom/torch_load_modelNvisualize.ipynb`
provides an example of plotting predictions.

## 8. Hyperparameter Tweaks

Experiment with convolution filter counts, kernel sizes or dropout rate.  Keep
the final output dimension at 303 so the `(3,101)` reshape remains valid.
