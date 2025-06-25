# Data Preprocessing Guide

This document explains how to generate the datasets used by the neural network models in this repository. The preprocessing notebooks were introduced over many commits and the latest workflow is summarized here.

## 1. Environment

Create a conda environment from `buIMU.yml` before running the notebooks:

```bash
conda env create -f preperation/buIMU.yml
conda activate buIMU
```

Python 3.8 with packages such as `numpy`, `pandas`, `scikit-learn` and `tensorflow` will be installed.

## 2. Preprocessing Steps

The pipeline expects raw IMU exports in text or CSV format. Execute the notebooks in the order listed below. Each step writes its results to the next folder so they can be reused without repeating the entire process.

1. **`0_Data_sorter.ipynb`** – Convert raw text files to CSV, normalize subject IDs and append side information. The cleaned files are organized into subject folders.
2. **`1_Data_Checker.ipynb`** – Split each CSV into individual gait cycles, discard truncated trials and update `list_dataset_correction.xlsx` with the valid recordings.
3. **`2_Data_PDFViewNCheck.py`** – Open the PDF plots for manual inspection. Press the indicated keys to mark trials as included (`IN`) or excluded (`EXC`). Selected files are copied to `Included_checked/RAW`.
4. **`3_0_Data_filtertoSave.ipynb`** – Apply a 12 Hz low‑pass Butterworth filter to the IMU columns (63 channels sampled at 250 Hz). Filtered CSVs are saved under `Included_checked/FILT` and summarized again in `list_dataset_correction.xlsx`.
5. **`3_1_Data_timenormalized.ipynb`** – Normalize each trial to 101 time points so that all gait cycles align.
6. **`3_2_Data_Exclusion.ipynb`** – Optional step to drop outlier trials using the criteria recorded in the `EXCLUSION_*` spreadsheets.
7. **`4_DataSet_CAN_MYWAY.ipynb`** – Standard pipeline combining the filtered CSVs, applying MinMax scaling and creating five cross‑validation folds.
   - **`4_DataSet_NAC_Mundt.ipynb`** – Reproduces the Mundt et al. scaling method for comparison purposes.
   - **`4_DataSet_IWALQQ_AE.ipynb`** – Writes the same trials into a four-dimensional tensor for autoencoder-based models.
   - **`4_DataSet_IWALQQ_AE_MOSTyle.ipynb`** – Formats the data to mimic the public MOST dataset.
   - **`4_DataSet_IWALQQ_AE_NOTSENSOR.ipynb`** – Drops selected sensors to evaluate robustness.
   Each notebook generates `<fold>_fold_final_train.npz` / `<fold>_fold_final_test.npz` and matching scaler pickles inside `preperation/SAVE_dataSet/<DATASET_NAME>` (e.g. `IWALQQ_1st_correction`).

### Choosing a Dataset

Most studies in this repository use **`4_DataSet_CAN_MYWAY.ipynb`** and keep
`nameDataset = "IWALQQ_1st_correction"`.  The NAC Mundt variant replicates the
scaling from Mundt&nbsp;et&nbsp;al., while the three `IWALQQ_AE*` notebooks
produce data formatted for autoencoders or sensor ablation experiments.  Pick the
notebook that matches your architecture and set the same `nameDataset` value
inside the training script.

Running these notebooks sequentially reproduces the dataset mentioned in the project README.

## 3. Dataset Layout

Each fold directory contains files structured as described below:

```
<SAVE_dataSet>/<DATASET_NAME>/
    0_fold_final_train.npz
    0_fold_final_test.npz
    0_fold_scaler4Y_angle.pkl
    0_fold_scaler4Y_moBWHT.pkl
    ... (repeat for folds 1–4)
```

The `.npz` archives store arrays named `final_X_train`, `final_X_test`, `final_Y_angle_train`, `final_Y_angle_test` and, if moment prediction is required, `final_Y_moBWHT_train` / `final_Y_moBWHT_test`.  Inputs have shape `(N, 4242, 1)` and targets `(N, 303, 1)` which represents `(3, 101)` time‑series data.

Scaler objects saved as pickle files are only used for computing metrics during training and evaluation.

## 4. Using the Datasets

Copy the generated `SAVE_dataSet` directory to a location referenced by the training scripts. By default they look for

```
preperation/SAVE_dataSet/<DATASET_NAME>/
```

You may override this path by editing the `absDataDir` variable inside the Python training files.

## 5. Architecture‑Specific Notes

All neural network variants load the same `.npz` and `.pkl` files but reshape the input tensors differently:

- **Dense models** (Keras and PyTorch) flatten each sample to `(4242,)` before feeding it to fully connected layers.
- **CNN models** expect data shaped `(42, 101)` so the channel dimension comes first.
- **LSTM and Transformer models** operate on sequences shaped `(101, 42)` with time as the leading dimension.
- **Demographic or autoencoder variants** load the same base arrays and concatenate additional features (e.g., age, sex, mass) from `demographics.xlsx` inside their custom `Dataset` classes.

The reshaping logic lives in the dataset classes.  Examples include
`Dataset4regressor` in `training/CBD/CBDtorch/custom/dataset.py`
(lines&nbsp;5–17) and the `Dataset` class inside
`training/MODEL/grid_torchDense/grid_torch_angleModel.py`
(lines&nbsp;84–92).  These classes read the `.npz` archives and use
`np.reshape` or `torch.squeeze` so the trainers can operate on the format
required by each architecture.

No additional preprocessing is required once the `.npz` archives and scaler pickles exist.
The resulting files can be used with every neural network architecture in this repository by specifying the appropriate dataset name when launching the training scripts.

The dataset class to load the archives is defined or imported near the top of each training script.  Look for a variable named `nameDataset` alongside a class such as `Dataset` or `Dataset4regressor`.  These classes reshape the arrays automatically, so you only need to adjust `nameDataset` to match the folder created by the preprocessing notebooks.  For instance, `torch_angleModel.py` defines `Dataset` at lines 81–95 while the `CBDtorch` models import `Dataset4regressor` from `custom/dataset.py`.

## 6. Tips and Troubleshooting

- Ensure that all paths in the notebooks match your local folder locations before executing a cell.
- If TensorBoard or training scripts cannot locate the scaler files, verify that their names match those written by the dataset notebook.
- Large intermediate CSVs and datasets are ignored by Git via `.gitignore`; keep them inside `preperation/` or another directory listed there.

After completing the above steps you can proceed to run any of the training scripts under `training/`.



