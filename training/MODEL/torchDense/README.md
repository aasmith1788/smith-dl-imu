# Dense_1st_torch Implementation Guide

This document explains how to reproduce the PyTorch-based multilayer perceptron (MLP) used in `torch_angleModel.py`. The model, referred to as **Dense_1st_torch**, predicts joint angles from IMU data. It is separate from the top-level README and focuses solely on this network.

## 1. Environment Setup

1. **Python version**: the scripts were tested with Python 3.8.
2. **Required packages**:
   - `torch` (tested with v1.12 or later)
   - `numpy`
   - `tqdm`
   - `tensorboard`
   - `pickle` (standard library)

Example installation with `pip`:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Format

Training and evaluation rely on five-fold cross validation. Each fold is stored as a compressed `.npz` archive containing arrays for the input and targets. The directory structure is:

```
<SAVE_dataSet>/<DATASET_NAME>/
    0_fold_final_train.npz
    0_fold_final_test.npz
    1_fold_final_train.npz
    1_fold_final_test.npz
    ... (up to 4_fold)
```

Within each archive:

- `final_X_<sess>` – IMU features for `train` or `test`
- `final_Y_angle_<sess>` – angle targets if `dataType` is set to `angle`
- optional arrays for other targets such as `final_Y_moBWHT_<sess>`

Input samples are shaped `(N, 4242, 1)` where `N` is the number of sequences in that fold. After squeezing the singleton dimension each sample becomes a `4242`‑long vector. Targets are stored as `(N, 303, 1)` which represents `(3, 101)` – three axes over 101 time steps. Scalar normalization parameters for each fold are saved alongside the datasets as pickle files:

```
<SAVE_dataSet>/<DATASET_NAME>/
    0_fold_scaler4Y_angle.pkl
    1_fold_scaler4Y_angle.pkl
    ...
```

These scalers are used only for computing metrics, not during forward passes.

## Data Preprocessing Pipeline (IWALQQ_1st_correction)

The `.npz` files and scalers originate from a series of notebooks under
`preperation/`.

1. **`0_Data_sorter.ipynb`** – Converts raw text exports from the motion capture
   system to CSV. File names are normalized and organized by subject.
2. **`1_Data_Checker.ipynb`** – Splits each CSV by step, removes truncated
   trials, and records which files are valid in `list_dataset_correction.xlsx`.
3. **`2_Data_PDFViewNCheck.py`** – Displays the accompanying PDF plots so each
   trial can be manually marked as included (`IN`), excluded (`EXC`) or
   postponed (`PP`). Chosen files are copied to `Included_checked/RAW`.
4. **`3_0_Data_filtertoSave.ipynb`** – Applies a 12&nbsp;Hz low-pass Butterworth
   filter to the IMU columns (63 sensor channels sampled at 250&nbsp;Hz). The
   filtered files are saved in `Included_checked/FILT` and summarized again in
   `list_dataset_correction.xlsx`.
5. **`4_DataSet_CAN_MYWAY.ipynb`** – Loads the filtered CSVs, generates features
   for each 101-sample gait cycle and performs a subject-level five-fold split
   using `KFold`. Inputs are scaled with `MinMaxScaler`, reshaped to
   `(N, 4242, 1)` and targets to `(N, 303, 1)`. Each fold is written to
   `preperation/SAVE_dataSet/IWALQQ_1st_correction/` as
   `{fold}_fold_final_train.npz` and `{fold}_fold_final_test.npz`. The fitted
   target scalers are stored alongside the archives as
   `{fold}_fold_scaler4Y_angle.pkl` (and similar files for moments).

Running these notebooks sequentially reproduces the dataset from the original
CSV exports. If you already have the `.npz` and `.pkl` files under
`preperation/SAVE_dataSet/IWALQQ_1st_correction/` there is no need to rerun the
pipeline.

## 3. Step-by-Step Training Guide

Follow these steps to train the network using the current repository layout.

1. **Prepare the dataset**
   - Create the directory `preperation/SAVE_dataSet/` at the project root.
   - Place the five fold archives under
     `preperation/SAVE_dataSet/IWALQQ_1st_correction/` so the script can find
     them.  Each archive must contain `final_X_train`, `final_X_test`,
     `final_Y_angle_train`, `final_Y_angle_test` and the matching scaler
     pickles.
2. **Edit `torch_angleModel.py`**
   - Set `nameDataset` to `"IWALQQ_1st_correction"` (or your folder name).
   - Set `dataType` to `"angle"` unless you wish to train on `moBWHT`.
   - `modelVersion` should remain `"Dense_1st_torch"`.
   - Adjust the directories `absDataDir`, `SaveDir` and `logDir`.  For a local
     run they can be relative, for example:

     ```python
     absDataDir = './preperation/SAVE_dataSet'
     SaveDir = './trainedModel'
     logDir = './training/logs/pytorch'
     ```

3. **Launch the training**
   - **Locally**: run `python training/MODEL/torchDense/torch_angleModel.py`.
   - **On BU SCC**: submit the batch script from inside
     `training/MODEL/torchDense/`:

     ```bash
     qsub ss_torch_angleModel.sh
     ```

4. **Check progress**
   - Job output on the cluster is written under `training/result_qsub/angle/`.
   - TensorBoard logs appear in the directory given by `logDir`, for example
     `training/logs/pytorch/<timestamp>/Dense_1st_torch/...`.
   - TorchScript checkpoints are stored in
     `trainedModel/Dense_1st_torch/<dataset>/<dataType>_<fold>_fold.pt`.

5. **Load the trained model**
   - Open `training/StudyRoom/torch_load_modelNvisualize.ipynb`.
   - Set the same paths used during training and run the notebook to load the
     `.pt` files and evaluate predictions.

---

## 4. Model Architecture

The MLP is implemented in `torch_angleModel.py` and consists of three fully connected layers with one dropout layer. Each batch element is flattened to a 1‑D vector of length 4242 before entering the network.

Layer by layer:

1. `Linear(4242, 6000)`
2. `ReLU`
3. `Dropout(p=0.5)`
4. `Linear(6000, 4000)`
5. `ReLU`
6. `Linear(4000, 303)`

The final layer outputs a vector of length 303 which corresponds to `(3, 101)` joint angles. No activation is applied to the output layer.

## 5. Forward Pass

During the forward pass each input tensor has shape `(batch, 4242)` after flattening. The network produces an output tensor of shape `(batch, 303)`. To compute normalized RMSE for each axis, the predictions and targets are reshaped inside the helper function `nRMSE_Axis_TLPerbatch`:

```python
pred_axis   = torch.transpose(torch.reshape(pred[bat], [3, -1]), 0, 1)[:, axis]
target_axis = torch.transpose(torch.reshape(target[bat], [3, -1]), 0, 1)[:, axis]
```

This converts the flat vector back to `(101, 3)` so that errors for X, Y and Z axes can be computed separately.

## 6. Loss Function

The training script uses a custom `RMSELoss`:

```python
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)
```

For reporting, the code calculates **normalized RMSE** (nRMSE) per axis. The predicted and true values are rescaled using the corresponding scaler for that fold:

```python
pred_axis = (pred_axis - scaler.min_[axis]) / scaler.scale_[axis]
```

nRMSE is then computed as:

```
100 * sqrt(mean((pred_axis - target_axis)^2)) / (max(target_axis) - min(target_axis))
```

## 7. Optimizer and Training Loop

- **Optimizer**: `torch.optim.NAdam`
- **Learning rate**: `0.0005`
- **Batch size**: `64`
- **Epochs**: `1000`

For each fold the script initializes a new model and trains with the dataset split for that fold. Training and test losses, along with per-axis nRMSE, are logged every epoch.

Backpropagation follows the standard PyTorch pattern:

```python
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

## 8. TensorBoard Logging

Two `SummaryWriter` instances record metrics for training and testing. Run TensorBoard pointing to the log directory to monitor the curves:

```bash
tensorboard --logdir <LOG_DIR>
```

The HParams tab will summarize the learning rate, batch size and other hyperparameters for each fold.

## 9. Saving and Loading the Model

After training a fold, the model is converted to TorchScript and saved:

```python
model_scripted = torch.jit.script(model)
model_scripted.save('.../Dense_1st_torch/<dataset>/<dataType>_<fold>_fold.pt')
```

Load the model later with:

```python
loaded = torch.jit.load('path/to/model.pt')
loaded.eval()  # Important before inference
```

## 10. Running Training Across Five Folds

1. Place the `.npz` files and scalers for all five folds under the dataset directory referenced by `dataSetDir` in the script.
2. Adjust the `totalFold`, `epochs`, or other settings at the top of `torch_angleModel.py` if desired.
3. Execute the script. It will loop over folds 0–4, training a new network for each and saving both TensorBoard logs and TorchScript checkpoints.

## 11. Modifying the Model

You can experiment by changing hidden sizes or dropout probability. For example, to reduce overfitting try lowering `Dropout(p=0.3)` or adding more layers. Keep the output dimension at 303 so the targets still reshape to `(3, 101)`.

---
This guide describes the **Dense_1st_torch** architecture exactly as implemented in `training/MODEL/torchDense/torch_angleModel.py`.
