# CNN_1st_torch Implementation Guide

This guide explains how to build and train the convolutional neural network referenced in
`torch_angleModel.py` when `modelVersion` is set to `"CNN_1st_torch"`.  The
workflow mirrors the dense model (see `Dense_1st_torch`) but replaces the MLP
with a small 1D CNN.  The same `.npz` datasets and `.pkl` scalers can be
reused without rerunning the preprocessing notebooks.

## Important Caveat

Reviewing the Git history shows that `torch_angleModel.py` never implemented a
convolutional model.  Every commit up through `df9474d` defines only the
`Mlp` class containing three dense layers.  Lines 90‑116 of the current script
illustrate this:

```python
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(4242, 6000)
        self.layer2 = nn.Linear(6000, 4000)
        self.layer3 = nn.Linear(4000, 303)
        self.dropout1 = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

Simply changing `modelVersion` therefore still trains this MLP.  To run a CNN
you must modify the script to include a convolutional architecture.

## History of CNN Experiments

Scanning every commit from the initial April&nbsp;2022 check‑in (`7095608`) to
the last refactor in `df9474d` shows only a few short trials with convolutional
layers.  Commits `106ab29` and `b4eaa03` introduced a file named
`tmp_torch.py`—a basic MNIST classifier with two `Conv2d` layers used merely to
verify GPU execution.  These experiments live under `training/StudyRoom` and
never connected to the IMU training scripts.  The notebook `QC_torch_angleModel`
added in `c07dd69` still created the dense `Mlp` model.  No commit defines a
`Conv1d` network for the IMU dataset or references the `CNN_1st_torch` version
flag.  The repository therefore contains no production CNN implementation nor

## History of LSTM Experiments

Several commits experimented with recurrent autoencoders rather than CNNs. The
one hundred ninth commit introduced `training/MODEL/Sequitur/LSTM_AE_1stTry.ipynb`
using the Sequitur library to build a simple `LSTM_AE` on one fold of the data.
The notebook ran a few forward passes but never reached a functional training
loop.

The next commit added a TensorFlow example under
`training/MODEL/Tensorflow_VAE_LSTM`. The accompanying
`VAE_TimeSeries.ipynb` notebook demonstrates a Keras-based VAE with LSTM layers
on a traffic-volume dataset to validate the approach before tackling the IMU
signals.

Commits one hundred eleven through one hundred fourteen then refactored the
PyTorch utilities into a small package named `CBDtorch` and created
`training/MODEL/Pytorch_AE_LSTM`. Inside `StudyRoom/AE_LSTM.ipynb` the authors
explored both unidirectional and bidirectional LSTM autoencoders. Later
revisions introduced a reusable `train_loop` routine and a companion notebook
showing how to reload a saved model.

Despite these investigations the main regression script never integrated an
LSTM. `torch_angleModel.py` still constructs only dense layers, so sequence
models remain confined to the experimental notebooks.

### Expansion of LSTM Work

Later commits moved beyond prototypes to a full VAE‑LSTM pipeline targeting the
knee OA dataset. Commit `2b6f5d4` (May 17 2022) imported a TensorFlow example to
study the architecture. Two days later `45c8935` introduced a PyTorch version
with notebooks illustrating both AE‑LSTM and VAE‑LSTM. Subsequent commits
`065e5d5` and `5eedc44` created the `CBDtorch` package with modular
`vaelstm_*layer.py` definitions so different encoder depths could be swapped in.
Training jobs under `training/MODEL/Pytorch_AE_LSTMwithDemographic` were
submitted to the BU SCC cluster via the `qsub_MOSTyle_torch_*` scripts. The
outputs—per‑epoch spreadsheets and summary PDFs—are stored beneath
`estimation/sensorwise/.../vaelstm_*`.

Commit `888428e` (June 28 2022) records running the third and fourth datasets
through this pipeline, and `187deff` (July 6 2022) began a variant using only
three sensors. These results demonstrate a working VAE‑LSTM model predicting
knee kinematics and kinetics, albeit outside of `torch_angleModel.py`.

## Adding CNN Support


Below is a minimal example of how `torch_angleModel.py` can be extended.  Add a
`CNN1st` class and instantiate it when `modelVersion` equals
`"CNN_1st_torch"`:

```python
class CNN1st(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(42, 64, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(32 * 93, 303)
    def forward(self, x):
        x = x.view(x.size(0), 42, 101)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        return self.fc(x)

if modelVersion == 'CNN_1st_torch':
    my_model = CNN1st()
else:
    my_model = Mlp()
my_model.to(device)
```

The rest of the training loop remains unchanged.

## 1. Environment Setup

1. **Python version**: Python 3.8 or newer.
2. **Required packages**:
   - `torch` (v1.12 or later)
   - `numpy`
   - `tqdm`
   - `tensorboard`
   - `pickle` (standard library)

Install them with `pip`:

```bash
pip install torch numpy tqdm tensorboard
```

## 2. Dataset Format

Training uses five-fold cross validation.  Each fold lives under
`preperation/SAVE_dataSet/<DATASET_NAME>/` as described in the dense model
README.  Every `.npz` archive contains `final_X_train`, `final_X_test` and the
matching targets (`final_Y_angle_*` or `final_Y_moBWHT_*`).  Input tensors are
stored as `(N, 4242, 1)` and targets as `(N, 303, 1)` which maps to `(3,101)`
angles or moments.  Each 4242-element vector actually represents 42 sensor
channels over 101 time steps.  For CNN training the data is reshaped to
`(N, 42, 101)` so that channel dimension is first.  This already matches
PyTorch's `Conv1d` expectation `(batch, channels, length)` so no `permute`
call is required.

Scaler pickles (`*_fold_scaler4Y_*.pkl`) are used only for metric
computations.

## 3. Training Steps

1. **Prepare the dataset**
   - Place all fold archives and their scalers under
     `preperation/SAVE_dataSet/<DATASET_NAME>/`.

2. **Configure `torch_angleModel.py`**
   - Set `modelVersion = "CNN_1st_torch"`.
   - Choose your dataset name (`nameDataset`), data type (`angle` or `moBWHT`),
     and absolute directories for `absDataDir`, `SaveDir` and `logDir`.
   - Example local setup:

```python
absDataDir = './preperation/SAVE_dataSet'
SaveDir   = './trainedModel'
logDir    = './training/logs/pytorch'
```

3. **Launch the training**
   - **Locally**: run

```bash
python training/MODEL/torchDense/torch_angleModel.py
```

   - **BU SCC**: submit `ss_torch_angleModel.sh` located in the same folder:

```bash
    qsub ss_torch_angleModel.sh
```

   The batch script loads the conda environment and allocates a GPU before
   executing the Python script.  The script itself detects the device via
   `torch.cuda.is_available()` and moves the model and data to it using
   `.to(device)`.

   Each of the five folds is trained sequentially.  After one fold completes a
   new model is initialized for the next fold, and its logs and checkpoint are
   stored in fold‑specific subdirectories.

## 4. Network Architecture

When `modelVersion` is `"CNN_1st_torch"` the model replaces the dense layers
with 1D convolutions.  The baseline configuration is:

1. `Conv1d(in_channels=42, out_channels=64, kernel_size=5, stride=1)`
2. `ReLU`
3. `Dropout(p=0.5)`
4. `Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1)`
5. `ReLU`
6. `Flatten`
7. `Linear(in_features=2976, out_features=303)`

The final linear layer outputs a length-303 vector which is reshaped to
`(3,101)` for metric calculations.  Adjust the convolution parameters if your
feature layout differs (time versus sensor axis).  With a kernel size of `5`
and no padding, the temporal dimension shrinks from `101` → `97` → `93`.  The
flatten operation therefore yields `32 × 93 = 2976` features.

## 5. Loss, Optimizer and Loop

The script defines a custom RMSE loss and computes normalized RMSE per axis as
shown around lines 102‑121 in `torch_angleModel.py`:

```python
pred_axis   = torch.transpose(torch.reshape(torch.squeeze(pred[bat]), [3,-1]), 0, 1)[:,axis]
```

The optimizer is `torch.optim.NAdam` with a learning rate of `0.0005`.
Training typically uses a batch size of `64` and `1000` epochs.  The standard
PyTorch loop updates the network and logs metrics every epoch.

## 6. Output and Visualization

TensorBoard writers store logs under
`<logDir>/<exp_name>/CNN_1st_torch/<dataset>/<dataType>/...` with a folder for
each fold.  Model checkpoints are saved as TorchScript modules in
`<SaveDir>/CNN_1st_torch/<dataset>/<dataType>_<fold>_fold.pt`.

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir <logDir>
```

## 7. Loading a Trained Model

Use `torch.jit.load` to load the scripted model:

```python
model = torch.jit.load('path/to/model.pt')
model.eval()
```

## 8. Tweaking Hyperparameters

Feel free to experiment with the number of convolutional filters, kernel sizes,
dropout rate, or learning rate.  Keep the final output dimension at 303 so that
targets reshape to `(3,101)`.

---
The preprocessing steps match those of **Dense_1st_torch**, so any `.npz` and
`.pkl` files created for that model can be reused directly.
