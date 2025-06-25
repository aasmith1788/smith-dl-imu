# ResNet_1st_torch Implementation Guide

This guide explains how to configure and train the **ResNet_1st_torch** model implemented in `torch_angleModel.py`. It assumes you already generated the five-fold dataset under `preperation/SAVE_dataSet/IWALQQ_1st_correction/` using the preprocessing notebooks.

## 1. Environment Setup

1. **Python**: version 3.8 or later.
2. **Packages**: `torch`, `numpy`, `tqdm`, `tensorboard`.

Install them with:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Format

Training relies on five-fold cross validation. Each fold is saved as an `.npz` archive containing `final_X_train`, `final_X_test`, and the matching targets. Inputs are stored as `(N, 4242, 1)` and targets as `(N, 303, 1)` – equivalent to `(3,101)` angles.
Scaler pickles (`*_fold_scaler4Y_*.pkl`) accompany each fold and are used for metric computation only.

The dataset structure mirrors that described in the **Dense_1st_torch** guide.

## Preprocessing Background

The `.npz` files were produced by a series of notebooks in `preperation/` which sort the raw CSV exports, filter the IMU signals, rescale them, and finally split subjects into five folds. If you already have the archives and scalers under `preperation/SAVE_dataSet/IWALQQ_1st_correction/` there is no need to rerun these notebooks.

## 3. Configuration Steps

Open `training/MODEL/torchDense/torch_angleModel.py` and edit the block around lines 16‑39 to configure paths and select the model version:

```python
exp_name = 'date_Dense_1st_torch'
modelVersion = 'Dense_1st_torch'
nameDataset = 'IWALQQ_1st_correction'
dataType = 'angle'  # or moBWHT
learningRate = 0.0005
batch_size = 64
```

Change `modelVersion` to `"ResNet_1st_torch"` and adjust `absDataDir`, `SaveDir`, and `logDir` as needed for your system.

## 4. Running the Training

1. **Prepare the dataset** – Place the `.npz` files and scaler pickles for all five folds under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.
2. **Launch training**
   - **Locally**: run `python training/MODEL/torchDense/torch_angleModel.py`.
   - **BU SCC**: submit the batch script `ss_torch_angleModel.sh` with `qsub`.

The script iterates over `totalFold=5`, initializing a new model for each fold. TensorBoard logs are created in the directory constructed at lines 144‑149 of the script.

## 5. Network Architecture

`ResNet_1st_torch` is a 1D residual network. After reshaping the input to `(batch, 42, 101)` the model processes it through three residual blocks:

1. **Block 1** – `Conv1d(42, 64, kernel_size=3)` → `BatchNorm1d` → `ReLU` → `Conv1d(64, 64, kernel_size=3)` → add skip connection.
2. **Block 2** – `Conv1d(64, 64, kernel_size=3)` → `BatchNorm1d` → `ReLU` → `Conv1d(64, 64, kernel_size=3)` → add skip.
3. **Block 3** – `Conv1d(64, 32, kernel_size=3)` → `BatchNorm1d` → `ReLU` → `Conv1d(32, 32, kernel_size=3)` → add skip.

After the residual stacks, features are flattened and passed through two dense layers:

4. `Linear(32 × 95, 512)` → `ReLU`
5. `Linear(512, 303)`

With no padding the temporal dimension decreases from 101 to 95. The output vector of length 303 is reshaped to `(3,101)` for loss and metric calculations.

## 6. Loss Function and Optimizer

The script defines a custom `RMSELoss` around lines 62‑70:

```python
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)
```

Optimization uses `torch.optim.NAdam` with a default learning rate of 0.0005. Training typically runs for 1000 epochs with a batch size of 64, but these values can be edited in the configuration block shown earlier.

## 7. Logging and Evaluation

Two `SummaryWriter` objects log metrics for each fold. Start TensorBoard pointing to `logDir` to monitor progress:

```bash
tensorboard --logdir <LOG_DIR>
```

Per-epoch losses and per-axis nRMSE are recorded, and the HParams tab summarizes the hyperparameters.

## 8. Model Saving and Loading

At the end of each fold the model is scripted and saved using TorchScript (lines 234‑237):

```python
dir_save_torch = join(SaveDir, modelVersion, nameDataset)
ensure_dir(dir_save_torch)
model_scripted = torch.jit.script(my_model)
model_scripted.save(join(dir_save_torch, f'{dataType}_{numFold}_fold.pt'))
```

Load a trained model later with:

```python
model = torch.jit.load('path/to/model.pt')
model.eval()
```

## 9. Five-Fold Loop

Set `totalFold = 5` to train sequentially on folds 0–4. The script reloads the dataset and creates fresh logs and checkpoints for each fold so results are kept separate.

## 10. Modifying the Model

Feel free to experiment with convolution filter counts, kernel sizes, or learning rate. Keep the final output dimension at 303 so predictions reshape to `(3,101)`.

---
This document should provide everything you need to train and evaluate **ResNet_1st_torch** using `torch_angleModel.py`.
