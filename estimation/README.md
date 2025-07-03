# Estimation

This folder holds notebooks and scripts used to generate knee angle and moment predictions from the trained networks.

## Model overview

Two separate models are built from the same IMU and demographic inputs:

1. **Kinematic model** – predicts knee joint angles.
2. **Kinetic model** – predicts knee moments.

A variational autoencoder (VAE) is first trained to learn a compact feature space. These VAE embeddings are then passed to fully connected layers to estimate the angles and moments. For comparison, a **reference model** implements only fully connected layers without the VAE stage, following the dense baselines used in previous studies.

### Reference dense scripts

The code for this reference approach lives under `training/MODEL/torchDense/`. Two
scripts mirror the VAE workflow but drop the autoencoder step:

- `torch_angleModel.py` – trains a multilayer perceptron to predict joint angles.
- `torch_momentModel.py` – trains the same network to estimate knee moments.

Both scripts use the same dataset splits and hyperparameters as the VAE version so
their results provide a direct baseline for comparison.

## Training procedure

All networks are trained with five‑fold cross‑validation. Early stopping monitors the validation error for ten epochs. Performance metrics include relative root‑mean‑square error (rRMSE in %) and correlation coefficients between predicted and reference measurements. For the kinetic model, correlations are also reported for the peak and impulse values of the knee flexion moment (KFM) and knee adduction moment (KAM).

## File layout

- **notsensor/** – Estimation notebooks that operate on the full set of sensors.
- **sensorwise/** – Utilities for experiments that evaluate each sensor individually.

Refer to the notebooks within these folders for examples of loading trained
models and creating summary tables. The dense reference model was evaluated
using the `sensorwise` notebooks, which save per-subject spreadsheets for each
sensor configuration. The `notsensor` notebooks merely gather those outputs to
produce the consolidated tables seen in the paper.
