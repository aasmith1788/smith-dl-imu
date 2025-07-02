# MODEL Directory Overview

The `MODEL` folder consolidates every training script developed for this repository. Early in the project the code base grew rapidly, scattering PyTorch and Keras experiments across multiple directories. Around the one hundred fifth commit the authors reorganized the entire repository so that all model implementations lived under `training/MODEL`. This change simplified job submissions on the BU SCC cluster and ensured that imports remained consistent no matter where the repository was cloned. What follows is a detailed description of each subfolder and the historical context that motivated its creation.

## kerasDense
The `kerasDense` folder houses the original multilayer perceptron regressors built with Keras. These models were the first successful baseline for estimating joint angles from IMU data. Each script defines a small fully connected network along with the job files used to launch training on the cluster. Although the project later migrated to PyTorch, these Keras scripts remain as a reference for the initial approach and as a comparison point for newer architectures.

## torchDense
This directory contains PyTorch implementations of the dense networks. The transition to PyTorch occurred as the team sought finer control over training loops and wanted to experiment with CNN, LSTM, and Transformer variants. Inside you will find separate README files describing each architecture, along with job scripts that mirror the Keras versions. Commit messages around the one hundred fiftieth revision note major refactors that unified output paths and added graph logging to these models.

## grid_torchDense
Hyperparameter sweeps for the PyTorch dense models live in `grid_torchDense`. Here the developers wrote shell scripts and small Python helpers that iterate over combinations of learning rates, batch sizes, and hidden layer counts. The grid searches produced tables of nRMSE values that guided architectural decisions. Historical commits show several adjustments to wall times and environment activation commands as the experiments matured.

## grid_torchDense_MOSTyle
When the project expanded to the MOSTyle dataset the team duplicated their grid search utilities to accommodate the new data format. This folder mirrors the structure of `grid_torchDense` but points to different dataset locations and saves logs with distinct prefixes. Commit messages mention the challenge of keeping track of multiple datasets and how this dedicated directory kept the runs organized.

## Pytorch_AE_LSTM
`Pytorch_AE_LSTM` marks the introduction of sequence autoencoders built with a small helper package called `CBDtorch`. Early commits packaged common dataset loaders, metric functions, and scaler utilities so they could be reused across notebooks. The experiments in this folder explore both vanilla LSTM autoencoders and variational counterparts. Many notebooks under `StudyRoom` demonstrate preliminary trials, while the Python modules implement reusable training loops.

## Pytorch_AE_LSTMwithDemographic
As the authors collected subject metadata they began adding age and sex inputs to the autoencoder. The resulting scripts live in `Pytorch_AE_LSTMwithDemographic`. Commit logs describe how the dataset preparation notebook was extended to store demographic arrays and how the training code was modified to concatenate these features with the IMU sequences. The team investigated whether demographic awareness improved reconstruction accuracy or introduced biases across the population.

## Pytorch_AE_LSTMwithMINI_Demographic
This variant uses a reduced demographic feature set to keep experiments lightweight. Historical notes show that the developers wanted a faster turnaround when sweeping hyperparameters, so they retained only a subset of the metadata. The folder otherwise mirrors the full demographic version in structure and job scripts.

## Pytorch_AE_MOSTylewithDemographic
When working with the MOSTyle dataset the team maintained demographic inputs, leading to this directory. The code adapts the loaders and models to the specific file layout of MOSTyle while keeping the same autoencoder architecture. Commit messages reference the difficulty of aligning subject identifiers between the primary dataset and the MOSTyle recordings.

## EXPERIMENTAL_Pytorch_AE_LSTMwithDemographic
Over time numerous exploratory notebooks accumulated, particularly for grid searches and ablation studies on the demographic models. These were collected under `EXPERIMENTAL_Pytorch_AE_LSTMwithDemographic`. The scripts in this folder produced hundreds of CSV files summarizing reconstruction error under different sensor combinations and latent dimensions. They serve as a record of the extended parameter sweeps that informed the final model choices.

## _NOTSENSOR
The `_NOTSENSOR` directory contains copies of many of the above scripts modified to intentionally remove one or more IMU channels. This line of research tested how well the models could recover missing data or operate with limited sensor coverage. According to later commit summaries, the authors duplicated several StudyRoom notebooks here to keep the altered experiments separate from the main workflow.

## Tensorflow_VAE_LSTM
Before standardizing on PyTorch the repository briefly explored a TensorFlow-based variational autoencoder. The notebook in this folder trains on a public traffic-volume dataset as a proof of concept. The experiment helped the team understand VAE behavior and informed the design of subsequent PyTorch implementations. Although it is mostly historical, the notebook provides a straightforward example of building a VAE with the Keras API.

## Sequitur
Finally, the `Sequitur` folder documents an attempt to use the Sequitur library for sequence autoencoders. The initial notebook loads the IMU dataset, constructs a simple `LSTM_AE` from Sequitur, and runs a few test passes. Commit notes reveal that the approach showed promise but ultimately failed to converge, so the experiment was shelved. The files remain for reference in case future work revisits the idea.

---

Together these folders chart the evolution of the project from simple dense networks to more complex autoencoder architectures. Reviewing the commit history alongside this overview should help new contributors understand how each experiment fits into the broader effort to model IMU data.
