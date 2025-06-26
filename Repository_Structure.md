# Repository Structure

This repository contains notebooks, scripts, and helper modules for training and evaluating deep learning models on inertial measurement unit (IMU) data. Preparation notebooks clean and organize the datasets, training scripts build various neural network models, and estimation notebooks analyze the resulting predictions.

## Folder Overview

- **preperation/** – Data preprocessing notebooks and utilities.
  - **StudyRoom/** – Exploratory notebooks for scaling methods and test set analysis.
  - **result_processingData/** – Outputs from processing pipelines such as normalized CSV files.
- **estimation/** – Notebooks and scripts that generate predictions from trained models.
  - **notsensor/** – Estimation notebooks for experiments without individual sensor analysis.
  - **sensorwise/** – Scripts and utilities for per-sensor evaluation.
    - Includes result folders (e.g., `Result_impulse`, `Result_peak`) and helper modules under `module/`.
- **training/** – Model definitions and training jobs.
  - **CBD/** – Custom package containing model classes like VAEs and dense networks.
  - **MODEL/** – Collection of training scripts organized by architecture (PyTorch and TensorFlow) and dataset variant.
  - **StudyRoom/** – Experiment notebooks and log folders for quick checks and debugging.

Model checkpoints and TensorBoard logs are saved to directories specified within the training scripts.
