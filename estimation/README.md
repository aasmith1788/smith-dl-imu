# Estimation Workflows

This folder contains the notebooks and scripts used after training to predict knee angles and moments from the recorded IMU signals. Everything here assumes that checkpoints already exist under `result_qsub/` or a similar directory. The contents fall into two complementary categories: **sensorwise** and **notsensor**.

## Sensorwise

The `sensorwise` subfolder focuses on evaluating one IMU location at a time. Notebooks such as `makeEstimationwithPDF_wDgMini.ipynb`, `makeEstimationwithPDF_wDgMOSTyle.ipynb`, and `makeEstimationwithPDF_woDg.ipynb` load a trained model and iterate through the dataset trial by trial. Each call to the network uses all 42 input channels—seven IMUs, two sensor types, and three axes—plus the demographic vector. The outputs for every trial are written as Excel files in directories named after the model variant.

Several helper scripts analyze these spreadsheets:

* `impulse_calculation.py` computes impulse metrics and stores them in `Result_impulse/`.
* `peak_detection.py` locates gait peaks and saves the results in `Result_peak/`.
* `plot_CBD.py` generates per‑axis plots for manual inspection.

Running a sensorwise notebook is computationally heavy because it loops over every trial and sometimes over multiple folds. By editing the data tensor before inference you can experiment with dropping or combining sensors, but you must handle the normalization yourself. The resulting Excel files remain compatible with the helper scripts, letting you compute errors and graphs for any configuration you choose.

## Notsensor

The `notsensor` folder serves two distinct purposes. One group of notebooks—including `maketable.ipynb` and `maketable_withDecay.ipynb`—merges the CSVs produced by the helper scripts into concise tables of relative RMSE and correlation. These aggregation notebooks run quickly and can be executed on any machine to regenerate summary statistics.

The second group evaluates alternate architectures that rely on all sensors simultaneously. Examples are `makeEstimationWithPDF_PyramidAttnCNNOPT.ipynb`, `makeEstimationWithPDF_DenseAngleArch.ipynb`, and `makeEstimationWithPDF_Optuna_1dCNN.ipynb`. They load their own checkpoints, run inference across the full dataset, and save predictions under `DenseModel/` using the same spreadsheet format as the sensorwise notebooks. Because the outputs look identical, you can merge these full‑sensor results with the sensorwise statistics using the same aggregation scripts.

These full‑sensor notebooks may give better overall accuracy because they exploit all available inputs and often employ more complex models. However, they do not reveal how each sensor contributes individually. Instead they provide a baseline of what is achievable when every IMU is active.

## Example workflow

1. Train the kinematic and kinetic networks under `training/`.
2. Open a sensorwise notebook, e.g. `makeEstimationwithPDF_wDgMini.ipynb`, and run all cells. Excel files for each trial appear in a folder like `wDgMiniModel/`.
3. Run `impulse_calculation.py` and `peak_detection.py` to transform the spreadsheets into concise CSVs within `Result_impulse/` and `Result_peak/`.
4. Use `notsensor/maketable.ipynb` to combine the CSVs into overall summary tables.
5. Optionally evaluate another architecture with a full‑sensor notebook such as `makeEstimationWithPDF_PyramidAttnCNNOPT.ipynb`. Its predictions go in `DenseModel/`, and rerunning `maketable.ipynb` will add them to the same tables.

## Choosing a workflow

Select **sensorwise** when you want detailed trial‑by‑trial predictions or when you need to test the effect of removing particular sensors. These notebooks require more processing time, but the fine‑grained outputs make it possible to examine individual IMU contributions and compute specialized metrics.

Select **notsensor** when you simply want summary tables or when you are benchmarking a model that uses every sensor simultaneously. Aggregation notebooks finish in minutes, and the full‑sensor inference scripts provide a quick baseline with minimal modification.

In practice you will often combine both approaches: run sensorwise notebooks to explore sensor importance or ablate inputs, then use the aggregation notebooks to collect the results alongside a full‑sensor architecture. Because every tool saves predictions in the same CSV format, the tables remain consistent regardless of which notebook produced them.
\nThe two approaches complement each other so you can draw conclusions about individual sensors while also benchmarking end-to-end accuracy.
