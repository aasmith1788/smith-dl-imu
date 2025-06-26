# Transformer_2nd_torch Implementation Guide

This guide describes how to train and evaluate the **Transformer_2nd_torch** network defined in `torch_angleModel.py`. It assumes the preprocessed dataset is available under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.

## 1. Environment Setup

1. **Python**: version 3.8 or newer
2. **Packages**: `torch`, `numpy`, `tqdm`, `tensorboard`

Install them with:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Requirements

Each of the five folds is stored as an `.npz` archive with arrays:

- `final_X_train` / `final_X_test` – input tensors shaped `(N, 4242, 1)`
- `final_Y_angle_train` / `final_Y_angle_test` – targets reshaped to `(3,101)` when loaded
- `*_fold_scaler4Y_angle.pkl` – scaler files used for metric computation only

During training the inputs are internally reshaped to `(N, 101, 42)` so that each sample represents 101 time steps with 42 features.

Place all files under `preperation/SAVE_dataSet/IWALQQ_1st_correction/`.

### Preprocessing Summary

The archives were produced by the notebooks in `preperation/` which converted raw CSV files, filtered and scaled the signals, and performed a five‑fold split. If the `.npz` and `.pkl` files already exist you can skip running the notebooks.

## 3. Configuration Steps

Open `training/MODEL/torchDense/torch_angleModel.py` and edit the settings block around lines 16‑39【F:training/MODEL/torchDense/torch_angleModel.py†L16-L39】:

```python
exp_name = 'date_Dense_1st_torch'
modelVersion = 'Dense_1st_torch'
nameDataset = 'IWALQQ_1st_correction'
dataType = 'angle'  # or moBWHT
learningRate = 0.0005
batch_size = 64
```

Change `modelVersion` to `"Transformer_2nd_torch"` and adjust `absDataDir`, `SaveDir`, and `logDir` for your environment. You may also modify `learningRate`, `batch_size`, `epochs`, or `totalFold`.

## 4. Running the Training

1. Verify the dataset files are placed as described above.
2. Launch training:
   - **Locally**: `python training/MODEL/torchDense/torch_angleModel.py`
   - **BU SCC**: submit `ss_torch_angleModel.sh` with `qsub`

TensorBoard logs are created in the directory specified in lines 145‑149 of the script【F:training/MODEL/torchDense/torch_angleModel.py†L145-L149】.

## 5. Model Architecture

`Transformer_2nd_torch` uses a deeper transformer encoder than the first version. After reshaping the input to `(batch, 101, 42)` the network performs the following steps:

1. **Embedding** – linear layer mapping 42 features to an embedding dimension (e.g., 128) with optional positional encoding
2. **Stack of TransformerEncoderLayer blocks** – typically four layers with multi‑head self‑attention (e.g., 8 heads) and feed‑forward sublayers
3. **Flatten** – the output sequence is flattened into a single vector
4. **MLP head** – `Linear(embed_dim × 101, 512)` → `ReLU` → `Linear(512, 303)`

The final output vector of length 303 is reshaped to `(3,101)` for loss calculation.

## 6. Loss and Optimizer

The script employs the custom `RMSELoss` defined around lines 62‑70 and optimizes the network using `torch.optim.NAdam`. Default hyperparameters are learning rate `0.0005`, batch size `64`, and epochs `1000`.

During each epoch the training and test sets are processed, and per‑axis normalized RMSE is logged to TensorBoard.

## 7. Logging and Model Saving

Summary writers create subfolders under `<logDir>/<exp_name>/Transformer_2nd_torch/...`. At the end of each fold a TorchScript model is written at lines 234‑237【F:training/MODEL/torchDense/torch_angleModel.py†L234-L237】.

View logs with:

```bash
tensorboard --logdir <logDir>
```

## 8. Loading a Trained Model

Load a saved model with:

```python
model = torch.jit.load('path/to/model.pt')
model.eval()
```

## 9. Five‑Fold Training Loop

The script iterates over `totalFold=5`, training a new model for each fold and storing logs and checkpoints separately.

## 10. Customization Tips

Experiment with embedding size, number of encoder layers, attention heads, or dropout rate. Keep the final output dimension at 303 so predictions reshape correctly to `(3,101)`.

---
This README provides all the information needed to train **Transformer_2nd_torch** using only `torch_angleModel.py` and the prepared dataset.
