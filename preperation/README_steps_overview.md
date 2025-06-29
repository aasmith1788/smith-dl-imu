## 0_Data_sorter.ipynb
- Scans the raw directory and groups every IMU CSV, marker trajectory file, force-plate TXT and PDF metadata sheet that share the same subject and trial ID into one folder.
- Renames each file to a uniform pattern (`IMU_TARGETLEG.csv`, `FORCE.txt`, `MARKER.trc`, etc.).
- Deletes the folder if any required component is missing and logs the omission.

## 1_Data_Checker.ipynb – integrity checks on the raw, still-separate sensor files inside each folder
- **Row-count parity:** compares row counts across the raw IMU, marker and force files; any mismatch of one or more rows rejects the trial.
- **Sampling interval (Δt) on raw IMU timestamps:** at a 100 Hz system (ideal Δt = 0.010 s), any gap outside 0.010 s ± 0.0005 s flags dropped or duplicated frames.
- **Monotonic timestamps:** raw IMU time column must strictly increase; repeats or back‑steps fail.
- **Channel count (IMU):** every raw IMU CSV must contain exactly 42 numeric columns (ACC X/Y/Z + GYRO X/Y/Z × 7 sensors).
- **Flat‑line detection (IMU):** variance of each IMU channel must exceed 1 × 10⁻⁶; if two or more channels are flat for the entire recording, the trial is rejected.
- **Signal range (IMU):** accelerometer values must stay within ±20 g and gyroscope within ±2000 deg·s⁻¹; more than 0.1 % out‑of‑range samples discards the trial.
- **NaN/Inf scan:** any non‑finite value, or more than 1 % of rows containing NaN/Inf in any file, triggers exclusion.

## 2_Data_PDFViewNCheck.py
- Auto‑plots each trial that passed the automated checks into a one‑page PDF.
- Lets the user tag the trial `IN`, `EXC` or `PP` via keyboard; the decision is saved to `list_Excluded_byFig.xlsx`.
- When run in EXP mode, copies all `IN` trials to `Included_checked/RAW` for downstream processing.

## 3_0_Data_filtertoSave.ipynb
- Loads each accepted trial's separate IMU, marker and force files, synchronises them on a common time axis and applies a 4th‑order Butterworth low‑pass filter (15 Hz accelerometer, 6 Hz moment).
- Removes spikes greater than 8 × MAD, reorders IMU channels, detects heel‑strike events and trims all signals to one complete gait cycle.
- Combines the aligned signals into unified NumPy arrays: IMU 101 × 42, knee angles 101 × 3, knee moments 101 × 3.

## 3_1_Data_timenormalized.ipynb
- Resamples every unified gait‑cycle array to 101 equally spaced points (0–100 % gait) and writes IMU, knee‑angle and knee‑moment matrices to disk.

## 4_DataSet_CAN_MYWAY.ipynb
- Splits the time‑normalised trials into five cross‑validation folds.
- Normalises knee moments by body‑weight × height, fits a MinMax scaler on each training split and scales all inputs and targets to the 0–1 range.
- Exports the fold‑wise NumPy archives and the six CSV files plus three plots used in your presentation.

![imu_acc_plot](https://github.com/user-attachments/assets/0433bc5c-2adb-4fad-a293-de2027930f1e)

![imu_gyro_plot](https://github.com/user-attachments/assets/842ae24d-0929-475c-af39-7ed1a77439cf)

![scaled_angle_plot_labeled](https://github.com/user-attachments/assets/4a53bcb2-ae87-407a-ae76-8a85fdf5fd5f)

![scaled_moment_plot_labeled](https://github.com/user-attachments/assets/7c0a8add-bba1-4855-8176-5426ef50e763)

