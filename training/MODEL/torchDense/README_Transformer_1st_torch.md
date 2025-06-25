# Transformer_1st_torch Implementation Guide

This guide explains how to configure, train, and evaluate the **Transformer_1st_torch** model in `torch_angleModel.py`. It assumes you have the preprocessed dataset under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.

## 1. Environment Setup

1. **Python**: version 3.8 or newer.
2. **Packages**: `torch`, `numpy`, `tqdm`, `tensorboard`.

Install them with:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Requirements

The training script expects five-fold archives created by the preprocessing notebooks. Each fold contains:

- `final_X_train` and `final_X_test` – input arrays shaped `(N, 4242, 1)`.
- `final_Y_angle_train` / `final_Y_angle_test` – targets reshaped to `(3, 101)` when loaded.
- `*_fold_scaler4Y_angle.pkl` – pickled scalers used only for computing metrics.

For the transformer model, inputs are reshaped inside the script to `(N, 101, 42)` so each sequence has 101 time steps with 42 features.

Place all `.npz` and `.pkl` files under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.

### Preprocessing Summary

The archives were generated via the notebooks in `preperation/` which sorted the raw CSV files, filtered and rescaled the signals, and performed a subject-level five-fold split. If these files already exist, you do not need to run the notebooks again.

## 3. Configuration Steps

Open `training/MODEL/torchDense/torch_angleModel.py` and edit the settings block around lines 16‑39【F:training/MODEL/torchDense/torch_angleModel.py†L16-L39】:

```python
exp_name = 'date_Dense_1st_torch'
modelVersion = 'Dense_1st_torch'
nameDataset = 'IWALQQ_1st_correction'
dataType = 'angle'
learningRate = 0.0005
batch_size = 64
```

Change `modelVersion` to `"Transformer_1st_torch"` and adjust `absDataDir`, `SaveDir`, and `logDir` for your system. You may also modify `learningRate`, `batch_size`, `epochs`, or `totalFold`.

## 4. Running the Training

1. Place the dataset files as described above.
2. Launch training:
   - **Locally**: `python training/MODEL/torchDense/torch_angleModel.py`
   - **BU SCC**: submit `ss_torch_angleModel.sh` with `qsub`.

TensorBoard writers are initialized around lines 145‑149 of the script【F:training/MODEL/torchDense/torch_angleModel.py†L145-L149】, placing logs in `<logDir>/<exp_name>/Transformer_1st_torch/<dataset>/...` for each fold.

## 5. Model Architecture

`Transformer_1st_torch` treats each input as a sequence of 101 steps with 42 features. A typical configuration is:

1. **Input embedding** – linear layer projecting 42 features to a higher dimension (e.g., 64), combined with positional encodings.
2. **Transformer encoder** – several stacked encoder blocks (`nn.TransformerEncoderLayer`) with multi-head self‑attention and feed‑forward layers.
3. **Flatten** the final sequence representation.
4. **MLP head** – `Linear(encoder_dim × 101, 512)` → `ReLU` → `Linear(512, 303)`.

The final 303-length vector is reshaped to `(3, 101)` when computing the loss.

## 6. Loss, Optimizer and Training Loop

The script defines a custom RMSE loss and uses `torch.optim.NAdam` with the learning rate specified in the configuration block. Training typically uses a batch size of 64 and runs for 1000 epochs, but these values are adjustable.

During each epoch the model processes the training and test sets, logging loss and per-axis normalized RMSE to TensorBoard.

## 7. Logging and Outputs

Logs are saved under the directory created by `SummaryWriter` calls at lines 145‑149. Model checkpoints are written as TorchScript files at the end of each fold (lines 234‑237)【F:training/MODEL/torchDense/torch_angleModel.py†L234-L237】.

View training progress with:

```bash
tensorboard --logdir <logDir>
```

## 8. Model Loading

Load a trained model with:

```python
model = torch.jit.load('path/to/model.pt')
model.eval()
```

## 9. Five-Fold Training Loop

The script iterates over `totalFold=5`, training a new model for each fold and storing logs and checkpoints separately.

## 10. Customization Tips

You can experiment with different embedding sizes, number of transformer layers, attention heads, dropout rate, or learning rate. Keep the final output dimension at 303 so predictions reshape to `(3, 101)`.

---
This guide provides everything needed to run **Transformer_1st_torch** using only `torch_angleModel.py` and the prepared dataset.

