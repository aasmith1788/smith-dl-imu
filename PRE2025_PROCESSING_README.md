# Data Processing Workflow Prior to 2025

This document summarizes how the repository's preprocessing notebooks cleaned the IMU data and where intermediate files were stored. It reflects the state of the codebase before any 2025 updates.

## Overview

1. **Sorting** (`0_Data_sorter.ipynb`) collects the raw IMU CSV files and renames them consistently.
2. **Initial Checks** (`1_Data_Checker.ipynb`) verifies row counts and timestamp integrity.
3. **Manual Review** (`2_Data_PDFViewNCheck.py`) plots each trial so users can mark problematic files.
4. **Filtering** (`3_0_Data_filtertoSave.ipynb`) synchronizes the IMU and force-plate data, applies a low-pass filter, and trims each recording to a single gait cycle. The output CSVs normally go in `preperation/NORM/`.
5. **Time Normalization** (`3_1_Data_timenormalized.ipynb`) resamples every cycle to 101 points.
6. **Optional Exclusion** (`3_2_Data_Exclusion.ipynb`) lets the user drop any trial that still looks questionable.
7. **Dataset Assembly** (`4_DataSet_CAN_MYWAY.ipynb`) reads `list_dataset_correction.xlsx` to determine which trials are valid, then bundles only those normalized CSVs into `.npz` archives and writes min–max scalers. The dataset folders live under `preperation/SAVE_dataSet/`.

## April 2022 Orientation Fix

On April 28, 2022 the team discovered that early recordings used inconsistent sensor orientations. They patched `0_Data_sorter.ipynb` and `1_Data_Checker.ipynb` to rotate each signal into the correct frame (commit `0f365875`). To avoid overwriting older results while validating the fix, `4_DataSet_CAN_MYWAY.ipynb` temporarily saved to `preperation/NORM_CORRECTION` with the dataset name `IWALQQ_1st_correction` (commit `f273d2b4`). Both `NORM` and `NORM_CORRECTION` were listed in `.gitignore` so intermediate files never entered version control.

The next day (commit `11079502`) the folder paths reverted to `preperation/NORM` and the dataset name became `IWALQQ_2nd_correction`. By that point the orientation fix was fully incorporated into the earlier notebooks, so the location change did not alter the data itself—only where it was written.

## Why It Matters

Rebuilding the pipeline from any commit after `0f365875` will produce properly aligned signals. If you run the notebooks at commit `f273d2b4`, the outputs appear under `NORM_CORRECTION`; if you run them at `11079502`, they go back under `NORM`. In both cases the `.npz` archives in `SAVE_dataSet/` contain the same corrected data. Later in 2025 the team continued using the `NORM` directory, which is why the current repository writes there.

## Reproducing the Corrected Dataset

1. `git checkout f273d2b4` (or `11079502` for the reverted paths)
2. Create the environment: `conda env create -f preperation/buIMU.yml && conda activate buIMU`
3. Run each notebook in numerical order from `0_Data_sorter.ipynb` through `4_DataSet_CAN_MYWAY.ipynb`
4. Find the results under `preperation/NORM_CORRECTION/` or `preperation/NORM/` depending on the commit

These steps regenerate the same files used for training in 2022, ensuring that the orientation fix is applied regardless of the output directory.
