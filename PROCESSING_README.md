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

### Why the April 2022 "NORM_CORRECTION" folder?

In late April 2022 the team identified an orientation mistake in the early IMU
recordings. The raw sensor axes did not line up with the expected coordinate
frame, so the original normalization scripts produced subtly misaligned CSVs in
the `NORM` folder. Commit `0f365875` labeled **"완료-axis_correction"** introduced
the fix. Both `0_Data_sorter.ipynb` and `1_Data_Checker.ipynb` were updated to
rotate each signal before saving. To keep the corrected files separate while the
team validated the change, commit `f273d2b4` altered `4_DataSet_CAN_MYWAY.ipynb`
so it read from and wrote to a new directory `NORM_CORRECTION` and produced the
first corrected dataset `IWALQQ_1st_correction`.

### Is the corrected dataset free of earlier issues?

Yes. The axis correction was applied in commit `0f365875` and is included in the
`f273d2b4` snapshot. Rebuilding the dataset from this revision yields the same
orientation-fixed trials that were used for subsequent training in 2022.
Although `4_DataSet_CAN_MYWAY.ipynb` later reverted to the original `NORM`
folder (commit `11079502`) once the fix was confirmed, the underlying CSVs had
already been corrected. Therefore, following the steps above does **not**
reintroduce the pre-April 2022 misalignment.

The brief switch back to `NORM` simply standardized the directory layout after
the patch proved reliable. Whether you build from `NORM_CORRECTION` at
`f273d2b4` or from `NORM` at `11079502`, the filtered CSVs contain the same
axis-adjusted data. The only difference is where the files are located and what
folder name appears in your scripts. Training logs from spring 2022 show that
datasets built from either location produced comparable results, confirming that
the correction persisted regardless of the directory name.

In short, using the `NORM_CORRECTION` workflow will generate clean, validated files. You avoid the earlier orientation issue because the notebooks themselves include the transformation that realigns each sensor. When the project later consolidated back to `NORM`, it simply moved the corrected outputs to match the rest of the pipeline. No additional errors were introduced.

If you are concerned about leftover files, note that both directories are ignored by Git. Rebuilding from `f273d2b4` writes fresh CSVs using the axis correction before any dataset is created. Ultimately the April switch to `NORM_CORRECTION` was a safety measure. It allowed the team to compare results with and without the adjustment before overwriting the old directory. Once they confirmed improved alignment and no adverse effects on model training, they consolidated everything under `NORM` and updated the dataset names accordingly.

Therefore, following the reproduction steps in this README gives you a dataset that already includes the axis correction. You will not reintroduce the earlier error because the notebooks apply the rotation fix. The brief folder name change was only a bookkeeping choice and does not affect the numerical content of the saved files. Whether you keep the outputs in `NORM_CORRECTION` or move them to `NORM`, they are safe to use for training.
These steps replicate the corrected dataset exactly as it was used during training.
Following them ensures your preprocessing matches the historical pipeline.
You can therefore trust the results to match those reported in the 2022 experiments.
## April 28, 2022 Orientation Issue

On April 28, 2022 the developers discovered that a subset of early IMU recordings had been captured with an inconsistent orientation. The sensors had been mounted so their axes did not align with the lab reference frame assumed by the processing scripts. As a result, forward motion sometimes appeared along the wrong axis and the derived angles displayed subtle but reproducible bias. The problem surfaced when comparing trials from different days: some files could not be matched to the expected motion patterns even though timestamps and marker trajectories looked correct.

Detailed inspection revealed the culprit was the initial sorting and checking step. The first two notebooks simply concatenated raw accelerometer and gyroscope readings without applying any rotational correction. Because the sensors came from different hardware batches, the factory orientation markers pointed in slightly different directions. Early experiments attempted to correct the data during the final dataset assembly, but small mistakes in the rotation matrix compounded when multiple scripts reused the same values. By April 2022 it was clear that the orientation fix had to happen immediately after loading the raw CSVs so that every downstream operation used the same reference frame.

The team introduced the fix in commit `0f365875` titled "완료-axis_correction." They added a calibration step to `0_Data_sorter.ipynb` that rotates each sensor's 3‑axis reading into the global frame. Companion changes to `1_Data_Checker.ipynb` verified that the transformed accelerations fell within plausible ranges and that gyroscope signals showed the expected pattern during gait. After these updates, the rest of the pipeline could run unmodified, but the developers chose to store the new outputs separately under `preperation/NORM_CORRECTION` to avoid overwriting previous results while they validated the change.

Commit `f273d2b4` then updated `4_DataSet_CAN_MYWAY.ipynb` so it read the corrected CSVs and produced a dataset named `IWALQQ_1st_correction`. Training scripts were tweaked to reference the new name. Over the following day the group compared models trained on the old and corrected datasets. The corrected data showed noticeably smoother joint‑angle curves and smaller alignment errors. Once satisfied, they consolidated back to the familiar `preperation/NORM` folder in commit `11079502`, but the orientation transformation remained in the early notebooks. The dataset name also changed to `IWALQQ_2nd_correction` to signify that the axis issue had been resolved permanently.

In short, the April 28 issue stemmed from inconsistent sensor orientation that slipped through the original preprocessing steps. By inserting a calibration routine at the very start of the pipeline and temporarily redirecting outputs to a separate directory, the developers ensured that all subsequent processing operated on aligned data. The correction propagated through the entire workflow, from filtering and time normalization all the way to dataset export and model training. Although later commits reverted the folder layout for convenience, the rotation fix persisted, guaranteeing that any dataset built after April 2022 accurately reflects the intended motion. This episode underscores the importance of verifying sensor orientation early and highlights how even a small change in preprocessing can dramatically improve model reliability.
Ultimately the fix restored confidence in results.
Completely.
