# Estimation

This folder collects every notebook used to evaluate trained networks on new IMU data. Two complementary workflows live here:

* **Sensorwise** notebooks load a checkpoint and **mask all but one sensor channel before predicting**, producing per-sensor metrics such as relative RMSE and correlation. Their outputs are saved to `Result_peak` and `Result_impulse` so you can examine how each individual IMU placement contributes to model accuracy.
* **Notsensor** notebooks either merge those CSV files into summary tables or run full-sensor evaluations with architectures like `makeEstimationWithPDF_PyramidAttnCNNOPT.ipynb`. The results share the same format, allowing all metrics to be combined later.

Although the trained models typically expect data from every sensor, isolating each location shows where the most useful information comes from and how performance changes if only a subset is available.

## Sensorwise notebooks

The `sensorwise` directory measures the impact of each IMU location. Notebooks such as `makeEstimationwithPDF_wDgMini.ipynb` load a checkpoint, mask all but one sensor channel, and then predict knee angles or moments. Each run writes relative RMSE and correlation values to `Result_peak` and `Result_impulse`. Utility scripts like `peak_detection.py` and `impulse_calculation.py` derive peak timing and impulse metrics from these files.

### Custom sensor setups

Every sensorwise notebook defines a variable called `mask` (or `active`) near its top. By default, the notebook loops over all sensors and sets this mask to enable one channel at a time. To examine a new combination, modify the mask list so that positions corresponding to desired sensors are set to `1`. Running the rest of the notebook will evaluate the model with only those channels enabled and save outputs in the same format. Because filenames and column layouts stay the same, the notsensor notebooks can merge these custom results without any further changes.

Helper modules under `sensorwise/module/` implement the moment, impulse, and peak calculations used by these scripts.

## Notsensor notebooks

The `notsensor` directory holds two kinds of notebooks. One group, exemplified by `maketable.ipynb`, simply merges the sensorwise spreadsheets to calculate average rRMSE and correlation. These summaries run in seconds. The other group performs full-sensor inference—`makeEstimationWithPDF_PyramidAttnCNNOPT.ipynb` is a common example—and saves its predictions under `DenseModel/` using the same CSV layout. All notsensor scripts rely on the files produced by the sensorwise notebooks so that results remain comparable.

### When should I run sensorwise?

Full-sensor notebooks such as `makeEstimationWithPDF_PyramidAttnCNNOPT.ipynb` provide the baseline performance when every IMU channel is available. Sensorwise notebooks are not redundant with this approach. They let you isolate each placement or experiment with subsets by modifying the `mask` variable. This is helpful when hardware constraints limit how many sensors you can mount, or when you want to verify that an individual device is reliable. Aggregation scripts in the notsensor folder then combine both the per-sensor and full-sensor results so that the tables reflect all of your experiments.
## Example workflow

1. Train the desired model in `training/MODEL/`.
2. Open `sensorwise/makeEstimationwithPDF_wDgMini.ipynb` and run all cells. This evaluates one sensor at a time and saves files under `Result_peak` and `Result_impulse`.
3. If you wish to test a pair of sensors, adjust the `mask` variable before running the notebook.
4. Run `sensorwise/peak_detection.py` or `sensorwise/impulse_calculation.py` if peak or impulse metrics are required.
5. Move to `notsensor/maketable.ipynb` to combine the spreadsheets into a single summary table.

Sensorwise analysis highlights which placements contribute the most. Notsensor merging turns those metrics into concise tables. Whenever you run a new configuration, save its spreadsheets in the usual folders and rerun the table notebook to update the results. This lightweight step ensures that all experiments remain comparable over time.
