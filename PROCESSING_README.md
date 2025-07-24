# Data Processing Overview

This document outlines the notebooks and scripts that clean the raw IMU dataset and produce the NumPy archives used for model training.  It reflects the pipeline as it existed in 2022 prior to the later 2025 updates.

## 1. Sorting and Integrity Checks

1. **`0_Data_sorter.ipynb`** – Gathers each subject's raw IMU CSV, marker trajectory file and force‑plate TXT into one folder and renames them consistently.
2. **`1_Data_Checker.ipynb`** – Verifies row counts, timestamp monotonicity and other quality checks on the raw files.
3. **`2_Data_PDFViewNCheck.py`** – Plots each trial to a PDF so the user can mark it as included or excluded.

## 2. Filtering and Normalization

4. **`3_0_Data_filtertoSave.ipynb`** – Synchronises IMU and force data, applies low‑pass filtering and trims each recording to one gait cycle.
5. **`3_1_Data_timenormalized.ipynb`** – Resamples every cycle to 101 points so all trials are time‑aligned.
6. **`3_2_Data_Exclusion.ipynb`** – Optional manual removal of problematic trials.

Intermediate results from these steps are written under `preperation/NORM/` by default.  The folder contains time‑normalized CSV files for each accepted trial.

## 3. Building the Dataset

7. **`4_DataSet_CAN_MYWAY.ipynb`** – Combines the normalized CSVs, splits them into five folds and exports `.npz` archives plus scalers.  Earlier commits saved to `preperation/NORM/`, but on 28&nbsp;April&nbsp;2022 the notebook was updated to write to a corrected directory `preperation/NORM_CORRECTION` and to produce a dataset named `IWALQQ_1st_correction`:

```python
nameDataset = "IWALQQ_1st_correction"
normalizedDir = join(dataDir, r'NORM')
...
normalizedDir = join(dataDir, r'NORM_CORRECTION')
```
Lines from the April commit show the dataset name and corrected path in the notebook:
【f273d2b4:preperation/4_DataSet_CAN_MYWAY.ipynb†L30-L40】【f273d2b4:preperation/4_DataSet_CAN_MYWAY.ipynb†L188-L205】

This change corresponded with the commit message "신규데이터추가" (`f273d2b4`, 28&nbsp;Apr&nbsp;2022).  The `.gitignore` was also updated so the new folder would not be tracked:

```text
preperation/NORM
preperation/NORM_CORRECTION
preperation/SAVE_dataSet
```
【F:./.gitignore†L1-L8】

8. **Training Scripts** – Subsequent training jobs loaded the corrected dataset as shown in `training/torch_angleModel.py` where `nameDataset` changed from `IWALQQ_1st` to `IWALQQ_1st_correction`.

```python
modelVersion = 'Dense_1st_torch'
nameDataset = 'IWALQQ_1st_correction'
```
【F:training/MODEL/torchDense/torch_angleModel.py†L48-L55】


Within a day the dataset builder switched back to the original folder. Commit `11079502` (29&nbsp;Apr&nbsp;2022) defines `nameDataset = "IWALQQ_2nd_correction"` and resets `normalizedDir` to `join(dataDir, r'NORM')`:

```python
nameDataset = "IWALQQ_2nd_correction"
normalizedDir = join(dataDir, r'NORM')
scalerDir = join(dataDir, r'SAVE_dataSet', nameDataset)
setDir = join(dataDir, r'SAVE_dataSet', nameDataset)
```
【F:preperation/4_DataSet_CAN_MYWAY.ipynb@11079502†L45-L60】【F:preperation/4_DataSet_CAN_MYWAY.ipynb@11079502†L136-L150】【F:preperation/4_DataSet_CAN_MYWAY.ipynb@11079502†L700-L745】
## Directory Summary

- **Raw data** – user supplied, not in the repository.
- **`preperation/NORM/`** – default location for filtered and time-normalized CSVs.
- **`preperation/NORM_CORRECTION/`** – temporary folder introduced in April 2022 for axis‑corrected files.
- **`preperation/SAVE_dataSet/`** – stores the final `.npz` and scaler files created by `4_DataSet_CAN_MYWAY.ipynb`.

Models trained with these datasets write logs under `training/logs/` and checkpoints in `training/result_qsub/` as configured inside each training script.

### Reproducing the corrected dataset

To rebuild the files under `NORM_CORRECTION`, check out commit `f273d2b4` and run the preprocessing notebooks in order. Then execute `4_DataSet_CAN_MYWAY.ipynb` from the same commit. The notebook will read `preperation/NORM_CORRECTION` and create `SAVE_dataSet/IWALQQ_1st_correction`.

### Step-by-step commands

```bash
git checkout f273d2b4
conda env create -f preperation/buIMU.yml
conda activate buIMU
jupyter notebook
```

Open `0_Data_sorter.ipynb` and progress sequentially through
`1_Data_Checker.ipynb`, `2_Data_PDFViewNCheck.py`,
`3_0_Data_filtertoSave.ipynb`, `3_1_Data_timenormalized.ipynb` and,
optionally, `3_2_Data_Exclusion.ipynb`.  Finally, run
`4_DataSet_CAN_MYWAY.ipynb` to generate the corrected dataset files under
`preperation/SAVE_dataSet/IWALQQ_1st_correction`.
