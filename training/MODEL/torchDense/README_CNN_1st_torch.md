# CNN_1st_torch Implementation Guide

This guide explains how to build and train the convolutional neural network referenced in
`torch_angleModel.py` when `modelVersion` is set to `"CNN_1st_torch"`.  The
workflow mirrors the dense model (see `Dense_1st_torch`) but replaces the MLP
with a small 1D CNN.  The same `.npz` datasets and `.pkl` scalers can be
reused without rerunning the preprocessing notebooks.

## 1. Environment Setup

1. **Python version**: Python 3.8 or newer.
2. **Required packages**:
   - `torch` (v1.12 or later)
   - `numpy`
   - `tqdm`
   - `tensorboard`
   - `pickle` (standard library)

Install them with `pip`:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Format

Training uses five-fold cross validation.  Each fold lives under
`preperation/SAVE_dataSet/<DATASET_NAME>/` as described in the dense model
README.  Every `.npz` archive contains `final_X_train`, `final_X_test` and the
matching targets (`final_Y_angle_*` or `final_Y_moBWHT_*`).  Input tensors are
stored as `(N, 4242, 1)` and targets as `(N, 303, 1)` which maps to `(3,101)`
angles or moments.  Each 4242-element vector actually represents 42 sensor
channels over 101 time steps.  For CNN training the data is reshaped to
`(N, 42, 101)` so that channel dimension is first.  This already matches
PyTorch's `Conv1d` expectation `(batch, channels, length)` so no `permute`
call is required.

Scaler pickles (`*_fold_scaler4Y_*.pkl`) are used only for metric
computations.

## 3. Training Steps

1. **Prepare the dataset**
   - Place all fold archives and their scalers under
     `preperation/SAVE_dataSet/<DATASET_NAME>/`.

2. **Configure `torch_angleModel.py`**
   - Set `modelVersion = "CNN_1st_torch"`.
   - Choose your dataset name (`nameDataset`), data type (`angle` or `moBWHT`),
     and absolute directories for `absDataDir`, `SaveDir` and `logDir`.
   - Example local setup:

```python
absDataDir = './preperation/SAVE_dataSet'
SaveDir   = './trainedModel'
logDir    = './training/logs/pytorch'
```

3. **Launch the training**
   - **Locally**: run

```bash
python training/MODEL/torchDense/torch_angleModel.py
```

   - **BU SCC**: submit `ss_torch_angleModel.sh` located in the same folder:

```bash
    qsub ss_torch_angleModel.sh
```

   The batch script loads the conda environment and allocates a GPU before
   executing the Python script.  The script itself detects the device via
   `torch.cuda.is_available()` and moves the model and data to it using
   `.to(device)`.

   Each of the five folds is trained sequentially.  After one fold completes a
   new model is initialized for the next fold, and its logs and checkpoint are
   stored in fold‑specific subdirectories.

## 4. Network Architecture

When `modelVersion` is `"CNN_1st_torch"` the model replaces the dense layers
with 1D convolutions.  The baseline configuration is:

1. `Conv1d(in_channels=42, out_channels=64, kernel_size=5, stride=1)`
2. `ReLU`
3. `Dropout(p=0.5)`
4. `Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1)`
5. `ReLU`
6. `Flatten`
7. `Linear(in_features=2976, out_features=303)`

The final linear layer outputs a length-303 vector which is reshaped to
`(3,101)` for metric calculations.  Adjust the convolution parameters if your
feature layout differs (time versus sensor axis).  With a kernel size of `5`
and no padding, the temporal dimension shrinks from `101` → `97` → `93`.  The
flatten operation therefore yields `32 × 93 = 2976` features.

## 5. Loss, Optimizer and Loop

The script defines a custom RMSE loss and computes normalized RMSE per axis as
shown around lines 102‑121 in `torch_angleModel.py`:

```python
pred_axis   = torch.transpose(torch.reshape(torch.squeeze(pred[bat]), [3,-1]), 0, 1)[:,axis]
```

The optimizer is `torch.optim.NAdam` with a learning rate of `0.0005`.
Training typically uses a batch size of `64` and `1000` epochs.  The standard
PyTorch loop updates the network and logs metrics every epoch.

## 6. Output and Visualization

TensorBoard writers store logs under
`<logDir>/<exp_name>/CNN_1st_torch/<dataset>/<dataType>/...` with a folder for
each fold.  Model checkpoints are saved as TorchScript modules in
`<SaveDir>/CNN_1st_torch/<dataset>/<dataType>_<fold>_fold.pt`.

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir <logDir>
```

## 7. Loading a Trained Model

Use `torch.jit.load` to load the scripted model:

```python
model = torch.jit.load('path/to/model.pt')
model.eval()
```

## 8. Tweaking Hyperparameters

Feel free to experiment with the number of convolutional filters, kernel sizes,
dropout rate, or learning rate.  Keep the final output dimension at 303 so that
targets reshape to `(3,101)`.

---
The preprocessing steps match those of **Dense_1st_torch**, so any `.npz` and
`.pkl` files created for that model can be reused directly.
