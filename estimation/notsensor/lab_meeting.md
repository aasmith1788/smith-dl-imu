![image](https://github.com/user-attachments/assets/2bad28d7-b4af-48a3-a239-9738273e2425)
![image](https://github.com/user-attachments/assets/3c514ca4-4d5e-4cb0-853e-0a976347f817)
![download](https://github.com/user-attachments/assets/f935df77-c27f-449a-b7e0-805f6aa8acdf)

![image](https://github.com/user-attachments/assets/869b839f-4cdd-4800-aac2-a5914b176cbc)
![image](https://github.com/user-attachments/assets/b1c396de-6e6f-45ab-a6bb-29ff19188d4e)
![download](https://github.com/user-attachments/assets/5cf7ee8d-4493-493d-8f9f-ec99b2159bbf)

The dense reference networks implemented in `torch_angleModel.py` and `torch_momentModel.py` share the same architecture. Each model is a three-layer multilayer perceptron (MLP). The input vector of 4,242 values is flattened and passed through two fully connected layers with 6,000 and 4,000 neurons respectively. A dropout layer with a rate of 0.5 follows the first layer to reduce overfitting. The final output layer contains 303 units to predict the sequence of joint angles or moments for all time steps.

Training scripts load the IMU features and target values from the five-fold datasets and train the MLP using either RMSE or MAE loss. Optimization is performed with the NAdam optimizer, and TensorBoard writers log per-epoch metrics for each fold. The same model class is used for both the kinematic (angle) and kinetic (moment) targets so differences in performance reflect the underlying data rather than the network structure.

The evaluation code reports **relative RMSE** (also called nRMSE) for each anatomical axis. During training this metric is computed by reshaping the model output and ground truth back to `(101, 3)` and undoing the scaling applied by the `MinMaxScaler` that was fitted on the training set.  For a single axis the routine loops over the batch, converts the values back to physical units using the stored `min_` and `scale_` parameters and then measures the root mean square error across all time steps.  Finally this error is divided by the dynamic range of the target motion:

```
nRMSE = 100 * sqrt(mean((pred - target)^2)) / (max(target) - min(target))
```
The `100` factor expresses the error as a percentage.  By normalizing with the range `(max - min)` the statistic indicates how large the error is relative to the subject's own movement amplitude.  This allows direct comparison across joints and participants even when their absolute joint angles or moments differ considerably, making nRMSE a consistent metric for training and reporting.

## Next Steps

While the dense MLPs provide a baseline, the repository is ready for more
expressive models.  Implementing a small **1D convolutional network** would
capture short-term temporal patterns that the current fully connected layers may
miss.  The existing dataset loaders can feed sequences of shape `(N, 4242, 1)`
into a series of convolutional and pooling layers before the final projection to
303 outputs.  See the `README_CNN_*` documents under
`training/MODEL/torchDense/` for preliminary design notes.

Another extension is a **temporal convolutional network (TCN)**.  By stacking
dilated convolutions with residual connections, a TCN can model long-range
dependencies while maintaining efficient training.  Adding such a model next to
the dense and CNN variants would enable direct comparison of how well each
captures movement dynamics across the five folds.
