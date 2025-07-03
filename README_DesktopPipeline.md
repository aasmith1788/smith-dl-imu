# Desktop Pipeline Guide

This short guide outlines the steps required to reproduce the IMU-based knee-angle and moment estimation pipeline using the scripts in this repository. It follows the workflow used to process the Boston University osteoarthritis dataset and train the VAE-LSTM models.

## 1. Acquire the dataset

The original study analysed 44 adults with medial knee osteoarthritis, yielding 876 valid walking trials. Ground-truth marker trajectories and force plates were recorded along with seven 9-axis IMUs. The raw files are not distributed here, so copy them to a local folder of your choice (for example `preperation/raw/`).

## 2. Preprocess the recordings

Run the notebooks under `preperation/` in the order below. They clean the raw files, apply filtering and create the fixed-length arrays used for training. A full description of each step appears in `preperation/README_steps_overview.md`.

1. `0_Data_sorter.ipynb`
2. `1_Data_Checker.ipynb`
3. `2_Data_PDFViewNCheck.py`
4. `3_0_Data_filtertoSave.ipynb`
5. `3_1_Data_timenormalized.ipynb`
6. `3_2_Data_Exclusion.ipynb` (optional)
7. `4_DataSet_IWALQQ_AE.ipynb`

Filtering is performed with a zero-phase Butterworth design. `README_steps_overview.md` notes a 4th-order low‑pass filter at 15 Hz for the accelerometer and 6 Hz for the moment channels:

```
21  - Loads each accepted trial's separate IMU, marker and force files, synchronises them on a common time axis and applies a 4th‑order Butterworth low‑pass filter (15 Hz accelerometer, 6 Hz moment).
```

Each stance phase is then resampled to 101 points and split into five cross‑validation folds.

## 3. Train the VAE-LSTM feature extractor

Edit the dataset and log paths near the top of `training/MODEL/Pytorch_AE_LSTM/torch_VAE_LSTM.py` and run the script:

```bash
python training/MODEL/Pytorch_AE_LSTM/torch_VAE_LSTM.py
```

## 4. Train the regression models

With the VAE trained, run the regression heads for angles and moments. The dataset name is defined on line 26 of `torch_regression_angle.py`:

```
26  nameDataset = "IWALQQ_AE_4th"
```

Execute the two scripts below (adjusting paths as needed):

```bash
python training/MODEL/Pytorch_AE_LSTM/torch_regression_angle.py
python training/MODEL/Pytorch_AE_LSTM/torch_regression_moBWHT.py
```

To include demographic features, use the versions under `Pytorch_AE_LSTMwithDemographic`:

```bash
python training/MODEL/Pytorch_AE_LSTMwithDemographic/torch_DG_regression_angle.py
python training/MODEL/Pytorch_AE_LSTMwithDemographic/torch_DG_regression_moBWHT.py
```

## 5. Evaluate the models

Open one of the notebooks in `estimation/notsensor/` or `estimation/sensorwise/` to compute metrics and produce the tables shown in the README. For instance run `makeEstimationwithPDF_wDg.ipynb` to evaluate the demographic model across all folds.

Model checkpoints are saved under the `trainedModel/` directory defined inside the scripts, while TensorBoard logs are written to `training/logs/`.

