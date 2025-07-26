# Estimation

This folder contains all notebooks for evaluating the trained networks on unseen IMU data. The code is organized into two complementary groups: `sensorwise` and `notsensor`. The former runs inference on one sensor at a time, while the latter collates those results into concise tables.

## Sensorwise notebooks

The `sensorwise` directory measures the impact of each IMU location. Notebooks such as `makeEstimationwithPDF_wDgMini.ipynb`, `makeEstimationwithPDF_wDgMOSTyle.ipynb`, and `makeEstimationwithPDF_woDg.ipynb` load a model checkpoint and mask all but one sensor channel before predicting. Each run writes an Excel file to `Result_peak` and `Result_impulse` capturing relative RMSE, correlation, and additional metrics. Helper scripts `peak_detection.py` and `impulse_calculation.py` compute peak timing and impulse values, while `plot_CBD.py` plots example gait cycles. The `_tensorboardResult` folder stores logs when these notebooks are executed on a GPU cluster.

### Custom sensor setups

Every sensorwise notebook defines a variable called `mask` (or `active`) near its top. By default, the notebook loops over all sensors and sets this mask to enable one channel at a time. To examine a new combination, modify the mask list so that positions corresponding to desired sensors are set to `1`. Running the rest of the notebook will evaluate the model with only those channels enabled and save outputs in the same format. Because filenames and column layouts stay the same, the notsensor notebooks can merge these custom results without any further changes.

Several helper modules under `sensorwise/module/` implement the shared metrics. For instance, `moment.py` and `impulse.py` integrate predicted moment curves, and `peak.py` finds maxima. The wrapper scripts `peak_detection.py` and `impulse_calculation.py` call these routines in parallel for different model folders. The resulting spreadsheets are placed in `Result_peak` or `Result_impulse` alongside the core prediction files.

## Notsensor notebooks

The notebooks in `notsensor` never run the models themselves. Instead they read the sensorwise outputs and produce overall statistics. The most important file is `maketable.ipynb`, which merges all `.xlsx` files from `Result_peak` and `Result_impulse` to compute the average performance of each sensor location for each model. Other notebooks, like `makeEstimationWithPDF_DenseAngles.ipynb` and `makeEstimationWithPDF_DenseMoments.ipynb`, generate the figures seen in presentations and papers. Because they operate only on stored results, they finish quickly even on a CPU.

The sensorwise directory also contains model subfolders such as `wDgModel`, `wDgMiniModel`, and `woDgModel`, which store checkpoints and scalers used by the notebooks. In parallel, the notsensor folder hosts notebooks like `makeEstimationWithPDF_PyramidAttnCNNOPT.ipynb` for experiments with a pyramid attention architecture. Keeping these files together ensures that every summary statistic can be traced back to the specific sensorwise predictions that produced it.
The subfolder `notsensor/DenseModel/` contains pre-generated outputs for the dense reference networks. You can inspect those results without first running the sensorwise notebooks. The additional `makeEstimationWithPDF_*` notebooks demonstrate alternative architectures or hyperparameters but still rely on the same merging routines.

## Example workflow

1. Train the desired model in `training/MODEL/`.
2. Open `sensorwise/makeEstimationwithPDF_wDgMini.ipynb` and run all cells. This evaluates one sensor at a time and saves files under `Result_peak` and `Result_impulse`.
3. If you wish to test a pair of sensors, adjust the `mask` variable before running the notebook.
4. Run `sensorwise/peak_detection.py` or `sensorwise/impulse_calculation.py` if peak or impulse metrics are required.
5. Move to `notsensor/maketable.ipynb` to combine the spreadsheets into a single summary table.

Sensorwise analysis exposes which placements contribute the most predictive power. Notsensor aggregation turns those detailed measurements into straightforward tables suitable for comparison across models. Whenever you customize the sensor mask or evaluate a new architecture, simply drop the resulting spreadsheets into the same folders. The notsensor notebooks will automatically incorporate them, ensuring that all reported averages correspond to the underlying sensorwise calculations. This separation keeps the evaluation pipeline both flexible and reproducible.
