# LSTM_1st_torch Implementation Guide

This guide describes how to train the **LSTM_1st_torch** version of `torch_angleModel.py`. It assumes you already have the preprocessed dataset under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.

## 1. Environment Setup

1. **Python**: 3.8 or later.
2. **Packages**: `torch`, `numpy`, `tqdm`, `tensorboard`.

Install with:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Format

The same five‑fold archives used for the dense and CNN models are required. Each fold contains arrays named `final_X_train`, `final_X_test` and corresponding targets. Inputs are `(N, 4242, 1)` and targets `(N, 303, 1)`【F:training/MODEL/torchDense/README_CNN_1st_torch.md†L25-L36】. For the LSTM model these inputs are reshaped to `(N, 101, 42)` so that each sequence has 101 time steps with 42 features per step.

Scaler pickles (`*_fold_scaler4Y_*.pkl`) are used for metric calculations only.

### Preprocessing Note

The `.npz` and `.pkl` files were generated using the notebooks in `preperation/` (sorting, filtering and rescaling the raw IMU CSVs). As long as those archives exist, no further preprocessing is needed.

## 3. Configuration Steps

Open `training/MODEL/torchDense/torch_angleModel.py` and edit the settings block around lines 16‑39 to select the LSTM model and set your paths:

```python
exp_name = 'date_Dense_1st_torch'
modelVersion = 'LSTM_1st_torch'
nameDataset = 'IWALQQ_1st_correction'
dataType = 'angle'  # or moBWHT
learningRate = 0.0005
batch_size = 64
```

Adjust `absDataDir`, `SaveDir` and `logDir` to match your system. Modify `epochs`, `totalFold` or the learning rate as desired.

## 4. Running the Training

1. Place all five `.npz` archives and scaler pickles under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.
2. Launch training:
   - **Locally**: `python training/MODEL/torchDense/torch_angleModel.py`
   - **BU SCC**: submit `ss_torch_angleModel.sh` with `qsub`.

TensorBoard writers are created at lines 144‑149 of the script【F:training/MODEL/torchDense/torch_angleModel.py†L144-L149】, placing logs in `<logDir>/<exp_name>/LSTM_1st_torch/<dataset>/...` for each fold.

## 5. Model Architecture

`LSTM_1st_torch` processes each input as a sequence of 101 steps with 42 features. A typical configuration is:

1. `nn.LSTM(input_size=42, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)`
2. Flatten the final hidden state (`64 × 101` → `6464`).
3. `nn.Linear(6464, 512)` → `ReLU`
4. `nn.Linear(512, 303)`

The 303‑length output is reshaped to `(3, 101)` when computing the loss.

## 6. Loss Function and Optimizer

A custom RMSE loss is defined in the script around lines 57‑70. Optimization uses `torch.optim.NAdam` with the learning rate specified above. Default batch size is 64 and training runs for 1000 epochs, but you may change these values in the configuration block.

## 7. Logging and Evaluation

Per‑epoch losses and normalized RMSE for each axis are logged to TensorBoard. Start TensorBoard pointing to your `logDir` to monitor progress:

```bash
tensorboard --logdir <LOG_DIR>
```

## 8. Model Saving and Loading

At the end of each fold the model is scripted and saved with TorchScript (lines 234‑237)【F:training/MODEL/torchDense/torch_angleModel.py†L234-L237】.

Load a trained model later with:

```python
model = torch.jit.load('path/to/model.pt')
model.eval()
```

## 9. Five‑Fold Loop

The script loops over `totalFold=5`, reloading the dataset and writing logs and checkpoints for each fold separately.

## 10. Modifying the Model

Experiment with different hidden sizes, number of LSTM layers or dropout rates. Keep the final output dimension at 303 so predictions reshape to `(3, 101)` correctly.

---
With this information you can train, evaluate and customize **LSTM_1st_torch** using only `torch_angleModel.py` and the preprocessed dataset.
