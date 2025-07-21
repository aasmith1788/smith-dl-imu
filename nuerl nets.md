# Naive Baseline Analysis Results

## Training-Mean Baseline Performance on Test Set

Take the mean of every trial at a normalized timepoint from the training set, that serves as your prediction for the test set. 

| Axis | Training Range | Global Correlation | Global RMSE | Global nRMSE | Mean Trial Correlation | Mean Trial nRMSE |
|------|---------------|-------------------|-------------|-------------|----------------------|-----------------|
| **X** | 75.98° | 0.793 | 6.299° | 8.29% | 0.941 ± 0.060 | 7.31% ± 3.92% |
| **Y** | 27.80° | 0.223 | 4.776° | 17.18% | 0.632 ± 0.299 | 14.57% ± 9.11% |
| **Z** | 55.54° | 0.254 | 5.910° | 10.64% | 0.499 ± 0.303 | 9.31% ± 5.16% |

_________________________________________________________________________________________________________________

Initial MLN


### Architecture Details

| Layer | Type | Input Size | Output Size | Activation | Notes |
|-------|------|------------|-------------|------------|--------|
| **Input** | Flatten | 4242 | 4242 | - | Converts input to 1D vector |
| **Layer 1** | Linear/Dense | 4242 | 6000 | ReLU | First hidden layer |
| **Dropout** | Dropout | 6000 | 6000 | - | 50% dropout rate for regularization |
| **Layer 2** | Linear/Dense | 6000 | 4000 | ReLU | Second hidden layer |
| **Output** | Linear/Dense | 4000 | 303 | None | Final prediction layer |

- **Loss Function**: RMSE 
- **Optimizer**: NAdam
- **Learning Rate**: 0.0005
- **Batch Size**: 64   -  looks at 64 training examples at once before updating its weights, rather than updating after every single example.
- **Cross-Validation**: 5-fold
- **Epochs**: 1000 per fold

### Test Preformance:

| Axis | Mean Correlation | Correlation Std | Mean nRMSE (%) | nRMSE Std (%) |
|------|-----------------|----------------|----------------|---------------|
| **X** | 0.936 | 0.068 | 7.77 | 4.11 |
| **Y** | 0.578 | 0.293 | 19.76 | 11.41 |
| **Z** | 0.353 | 0.338 | 11.42 | 6.61 |

Axis X: mean SD ratio = 0.384
Axis Y: mean SD ratio = 0.350
Axis Z: mean SD ratio = 0.607

### Training Preformance


# Dense MLP Training Results (Population Range-based)

## Training Performance Summary (5-fold Cross-Validation)

| Axis | Population Range | Mean Correlation | Correlation Std | Mean Relative RMSE (%) | RMSE Std (%) |
|------|-----------------|-----------------|----------------|----------------------|---------------|
| **X** | 75.98° (-67.41° to 8.56°) | 0.937 | 0.066 | 8.03 | 4.09 |
| **Y** | 27.80° (-13.18° to 14.61°) | 0.602 | 0.292 | 18.75 | 9.88 |
| **Z** | 55.54° (-41.26° to 14.28°) | 0.420 | 0.345 | 10.30 | 5.17 |

Axis X: mean SD ratio = 0.927
Axis Y: mean SD ratio = 0.669
Axis Z: mean SD ratio = 0.532

_____________________________________________________________________________________________________________________________

### OPTIMIZED MLN

 - KEY CHANCGES
 - Batch Normalization: Added after each hidden layer for stable training
 - Early Stopping: Stops training when validation loss plateaus (patience=10)
 - Learning Rate Scheduling: Reduces LR by 50% when validation stagnates
 - He Weight Initialization: Better initial weights for ReLU networks
 
# Test Preformance

| Axis | Population Range | Mean Correlation | Correlation Std | Mean Relative RMSE (%) | RMSE Std (%) |
|------|-----------------|-----------------|----------------|----------------------|---------------|
| **X** | 75.98° (-67.41° to 8.56°) | 0.920 | 0.064 | 10.62 | 7.58 |
| **Y** | 27.80° (-13.18° to 14.61°) | 0.515 | 0.263 | 14.11 | 7.93 |
| **Z** | 55.54° (-41.26° to 14.28°) | 0.440 | 0.279 | 12.85 | 7.79 |

Axis X: mean SD ratio = 1.175
Axis Y: mean SD ratio = 0.703
Axis Z: mean SD ratio = 0.96


# Train Preformance

| Axis | Population Range | Mean Correlation | Correlation Std | Mean Relative RMSE (%) | RMSE Std (%) |
|------|-----------------|-----------------|----------------|----------------------|---------------|
| **X** | 75.98° (-67.41° to 8.56°) | 0.989 | 0.016 | 2.85 | 1.63 |
| **Y** | 27.80° (-13.18° to 14.61°) | 0.917 | 0.100 | 3.18 | 2.01 |
| **Z** | 55.54° (-41.26° to 14.28°) | 0.888 | 0.144 | 3.24 | 1.79 |

Axis X: mean SD ratio = 1.003
Axis Y: mean SD ratio = 0.882
Axis Z: mean SD ratio = 0.849
