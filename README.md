# Smith DL IMU

This repository explores deep learning methods for predicting biomechanical
quantities from inertial measurement unit (IMU) recordings.  What follows is a
detailed description of what the codebase looked like at the very beginning of
the project.

## Summary of the Initial Commit

The very first commit introduced a set of prototype training programs written in
Python.  Two key scripts (`batch_5fold_angleModel.py` and
`batch_5fold_momentModel.py`) drove five‑fold cross‑validation experiments using
TensorFlow/Keras.  Each constructed a simple sequential neural network, loaded
pre‑scaled data from `DATASET/` and `SCALER/`, and measured performance with
custom metrics like rescaled root‑mean‑square error for each axis of the target
variables.

Running these experiments on the Boston University SCC high‑performance
computing cluster required dedicated job scripts.  `angleModel.sh` and
`momentModel.sh` specified SGE queue options, loaded a conda environment named
`sccIMU`, configured TensorFlow thread counts, and finally launched the Python
trainers.  Their presence shows that the project was intended to scale beyond a
local workstation from the outset.

Complementing the batch scripts was a Jupyter notebook checkpoint,
`5_FirstModel-checkpoint.ipynb`, which contained exploratory TensorFlow code and
notes.  A large job log (`thirdTry_moment.o4227182`) recorded the output of a
previous cluster run and serves as evidence of early experimentation.  An empty
placeholder file with a similar name was committed inadvertently.  A simple
`.gitignore` rounded out the initial snapshot, excluding local data, logs,
saved models and scaler objects from version control.

## What the First Experiment Did

The researchers attempted to predict lower‑limb joint angles and moments from
time‑normalized IMU signals.  Each fold of the dataset contained 4,242 features
representing sensor readings across the gait cycle.  The neural network mapped
these inputs to 303 target values for three anatomical axes and was trained in a
five‑fold cross‑validation loop.

The model architecture comprised two dense layers with 6,000 and 4,000 units,
respectively, followed by dropout and a 303‑unit output layer.  Custom metrics
such as the rescaled root‑mean‑square error transformed predictions back to the
original units using stored scalers.  The scripts ran for up to 1,000 epochs on
the BU SCC cluster using the Nadam optimizer and logged metrics via TensorBoard.

In this fully connected neural network architecture, information flows through each layer where every unit receives all outputs from the previous layer and produces a single output value. Starting with 4,242 IMU features, each of the 6,000 units in the first hidden layer receives all 4,242 inputs, applies its own unique weights and bias, and outputs one value through an activation function, producing 6,000 total outputs. These 6,000 values then feed into the second hidden layer where each of the 4,000 units receives all 6,000 inputs, applies its own learned weights and bias, and produces one output value, resulting in 4,000 total outputs. Finally, each of the 303 output units receives all 4,000 values from the previous layer, applies its own weights and bias, and produces one final prediction corresponding to specific joint angles or moments across the three anatomical axes. This "fully connected" or "dense" design means every unit in each layer connects to every unit in the previous layer with its own learnable weight, allowing the network to discover complex combinations of features that are most predictive for biomechanical estimation while progressively transforming the high-dimensional sensor data into the final 303 biomechanical predictions.

## Reproducing the Original Training Runs

To recreate the first experiments, begin by installing Python 3.8+ and setting up a virtual environment.  The early scripts relied on TensorFlow 2.x, Keras, scikit-learn, and numpy.  After cloning this repository, install these packages using `pip` or recreate the `sccIMU` conda environment referenced in the job scripts.  Running on a GPU is optional but provides substantial speedups.

Next, gather the preprocessed data expected by the trainers.  The initial commit assumed that each fold of the five‑fold cross‑validation was saved in `DATASET/` as NumPy `.npz` archives, while matching scaler objects lived in `SCALER/`.  These files can be produced from raw IMU measurements using your own preprocessing pipeline or obtained from the project authors.  Ensure that every fold includes `final_X_train`, `final_X_test`, and the corresponding target arrays along with three scaler pickle files.

With the environment and data in place you can run `batch_5fold_angleModel.py` or `batch_5fold_momentModel.py` directly to execute all five folds locally.  On clusters using Sun Grid Engine, submit `angleModel.sh` or `momentModel.sh` to queue the jobs.  Each fold writes TensorBoard logs under `logs/fit/` and prints metrics such as the rescaled RMSE for each axis.  Inspect these logs to evaluate model performance and compare against the original results.

## Summary of the Second Commit

The second commit turned attention toward verifying the accuracy of the custom
metrics used during training.  A new notebook named
`StudyRoom/Check_metric.ipynb` was introduced to carefully inspect model
predictions.  Within this Jupyter environment the developers loaded previously
computed outputs for each fold, applied the stored scalers to convert them back
into real-world units, and then manually calculated the per-axis RMSE.  By
walking through these steps interactively they could confirm that the formulas
embedded in the training scripts produced sensible numbers.

To better organize saved models, `batch_5fold_momentModel.py` gained a
`modelVersion` variable that is appended to each HDF5 filename.  During this
phase every fold wrote its weights to the new `SavedModel/` directory using
`model.save()`.  The `.gitignore` file was extended so these binary snapshots did
not clutter version control.  Running the updated script on the BU SCC generated
`moment.o4238965`, while the prior log `thirdTry_moment.o4227182` was renamed to
`thirdTry_angle.o4227182` so it clearly referred to the angle model.  A small
placeholder file, `thirdTry_moment.po4227182`, was accidentally included as
well.

## What the Metric Experiment Did

Although the network architecture itself remained unchanged, this phase was all
about confirming that the metric calculations were correct.  By recomputing the
errors outside the training loop, the team ensured that rescaling operations and
axis-wise reductions matched their expectations.  The saved weights meant future
analyses could be performed without rerunning costly training jobs, and the new
log provides a point of comparison for subsequent improvements.

## Reproducing the Metric Validation Stage

Begin by ensuring the same Python environment from the first commit is active
and that a `SavedModel/` folder exists.  Running
`batch_5fold_momentModel.py` will once again process each fold but now produces
a uniquely named HDF5 file for every run.  After training finishes check that
`moment.o4238965` appears and that the saved models reside under
`SavedModel/`.

Launch Jupyter and open `StudyRoom/Check_metric.ipynb`.  Execute the notebook
cell by cell so it loads your prediction files, rescales them and computes the
per-axis RMSE.  If the values printed here match those in the log you have
successfully reproduced the authors' validation process.

Finally, inspect the generated models or plug them into your own evaluation
scripts.  Because the commit's goal was metric verification, these artifacts are
the baseline for later iterations and should remain unchanged unless you modify
the preprocessing pipeline.

## Summary of the Third Commit

The third commit expanded the metric study by heavily revising
`StudyRoom/Check_metric.ipynb`.  Dozens of existing cells were cleaned up and
many new ones added so that predictions from each fold could be loaded and
reshaped in several different ways.  These edits helped the team reason about
how TensorFlow arranged tensors internally and ensured that the RMSE was always
computed over the correct axis.  A fresh training run captured its console
output in `moment.o4238965` while a zero-byte placeholder log named
`moment.po4238965` appeared as well.

Because this commit focused solely on the moment model, no changes were made to
the angle scripts.  Instead the notebook became the central place to verify that
metrics computed inside the training loop were identical to those calculated
after the fact.  The large diff in the notebook history shows repeated attempts
to visualize intermediate tensors and to understand the effect of transposing
data before applying scaler objects.

## What the Notebook Exploration Did

By iterating through multiple data layouts, the team confirmed that transposing
the prediction tensor prior to scaling gave consistent RMSE values across all
axes.  Visualizations embedded in the notebook plot the predicted moment curves
against the ground truth, making it easy to spot any misalignment.  The new log
file captures a run where these reshaping steps were applied, serving as proof
that the corrections functioned in practice.

Commit three focused exclusively on debugging the moment model's metric calculations through extensive experimentation in the validation notebook, which became the central testing ground for ensuring consistency between training-time and post-training RMSE calculations. The researchers needed to verify that when their training loop reported metrics like "X-axis RMSE = 5.2," manually recalculating the same RMSE in the notebook using saved predictions would yield identical results, as any discrepancy would indicate a fundamental error in their metric implementation. The commit shows evidence of intensive detective work through numerous notebook modifications - adding, deleting, and revising cells as they experimented with different approaches to visualize tensor shapes, test various transposition and reshaping strategies, and determine the correct sequence of operations (whether to transpose before scaling or scale before transposing). This iterative trial-and-error process in the notebook allowed them to systematically debug their data processing pipeline and ensure their per-axis RMSE calculations were mathematically sound and reproducible across both the automated training environment and manual verification steps.

## Reproducing the Notebook Improvements

First, open `StudyRoom/Check_metric.ipynb` in Jupyter and execute it from the
top.  The notebook expects the HDF5 models saved by the previous commit, along
with their prediction outputs.  Verify that each cell runs without error and that
the plots of predicted versus true moments appear as shown in the repository.

Next, examine the tensor shapes printed in the early cells.  These should match
the dimensions documented in the markdown comments.  If they do, continue to the
RMSE calculation section and confirm that the error metrics agree with those in
`moment.o4238965`.

Finally, submit `momentModel.sh` or run `batch_5fold_momentModel.py` locally to
produce a fresh log.  Compare this output with the one committed in the repo to
ensure your environment replicates the original experiment.

## Summary of the Fourth Commit

Next, attention shifted back to the training scripts themselves.  Both
`batch_5fold_angleModel.py` and `batch_5fold_momentModel.py` were refactored so
their custom metric functions first transpose the prediction tensor.  This
ensures each anatomical axis is evaluated independently before the values are
rescaled to physical units.  The notebook received matching updates and several
redundant code cells were removed to keep it aligned with the streamlined
functions.

## What the Axis Refactor Did

Prior to this refactor the metrics occasionally mixed axes because the output was
scaled before being rearranged.  By explicitly transposing first, the code now
computes the RMSE for X, Y, and Z separately and then averages them.  This
provides more interpretable feedback and prevents subtle broadcasting bugs that
could hide poor performance.

## Reproducing the Axis Refactor Stage

Run either training script after pulling this commit to generate new results.
Watch the console output to see three separate RMSE values per axis.  After the
jobs finish, open `StudyRoom/Check_metric.ipynb` and confirm that running its
cells reproduces the exact same errors using the saved predictions.

If your metrics disagree, double-check that the scalers in `SCALER/` correspond
to the dataset used for training.  Mismatched scalers will lead to inconsistent
results.

## Summary of the Fifth Commit

Finally, a short follow-up commit captured one more training run and tidied up
the metric implementation.  A new log file `angle.o4249389` records the output
from submitting the angle-model job on the cluster.  Meanwhile
`batch_5fold_momentModel.py` adjusted its `rescaled_RMSE_pct` helper so that it
operates on the entire transposed tensor rather than indexing into a single
slice.  This prevents out-of-bounds errors when evaluating multiple axes.

## What the Error Fix Provided

The additional log demonstrates that the training loop now runs smoothly with no
indexing issues.  Loss curves decline steadily and the per-axis RMSE values are
comparable to earlier runs, confirming that the fix did not change the overall
behavior.  With the metric code stable the researchers could focus on more
ambitious experiments.

## Reproducing the Fix and Log Generation

Run `batch_5fold_momentModel.py` once more to verify that the updated metric no
longer triggers index errors.  Inspect the console output or log file to confirm
three RMSE values are printed for each fold.

Next, submit `angleModel.sh` on your cluster.  It should produce a log named
`angle.o4249389` that mirrors the one in the repository.  Review this file to
ensure training proceeds through all epochs without warnings.

These steps establish a clean baseline from which later commits will explore
different architectures and datasets.


## Summary of the Sixth Commit

The sixth commit concentrated on verifying that the earlier metric fixes worked for every output dimension.  The notebook `StudyRoom/Check_metric.ipynb` gained new cells that load the prediction arrays in one go, transpose them so the axes line up, and then apply the scalers across the entire matrix.  Instead of looping through each axis separately, the revised code performs vectorized calculations so that numerical precision matches the training scripts exactly.  A new cluster log, `angle.o4249389`, captures the output from running the angle model with these updates.

## What the Final Verification Did

Executing the improved notebook demonstrated that rescaled RMSE values remained consistent across all three anatomical axes.  The log file shows the same error reported in the notebook, proving that the training scripts and manual calculations now agree.  By ensuring the metric function handled the whole tensor at once, the team eliminated corner cases where slices might be misaligned or scaled twice.

## Reproducing the Final Verification

Launch Jupyter and step through `StudyRoom/Check_metric.ipynb` after training the model with `batch_5fold_angleModel.py`.  Pay close attention to the shapes printed by each cell—they should display `(3, N)` after the transpose, where `N` is the number of time points.  When the notebook completes, submit `angleModel.sh` on your cluster and compare the resulting `angle.o4249389` with the repository’s copy.  Matching RMSE values confirm the fix.

## Summary of the Seventh Commit

With the metrics validated, the seventh commit tidied up the repository.  The development notebook `Check_metric.ipynb` was pruned of experimental cells and reorganized so the important plots appear at the end.  An enormous earlier log file, `thirdTry_angle.o4227182`, was deleted because its thousands of lines made diffs unwieldy.  The team then reran the angle model, producing a fresh record named `angle.o4249548`; the cluster also created a zero-byte `angle.po4249548` placeholder during job submission.

## What the Metric Study Added

The streamlined notebook now plots predicted angles versus ground truth for each fold and lists the exact RMSE values underneath.  These visuals make it clear how well the model tracks joint motion throughout the gait cycle.  By trimming the obsolete log the commit also reduces repository size, easing future cloning and review.

## Reproducing the Metric Study

Execute `angleModel.sh` to generate new logs in your environment.  Afterwards launch `Check_metric.ipynb` and run through all cells.  Inspect the final plots to confirm they resemble those stored in version control and check that `angle.o4249548` matches the published values.  If your results differ significantly ensure that the scaler files correspond to the dataset used for training.

## Summary of the Eighth Commit

After reviewing the previous results the researchers concluded that averaging the RMSE across axes obscured important differences in performance.  They therefore removed the `rescaled_RMSE` metric from both `batch_5fold_angleModel.py` and `batch_5fold_momentModel.py`.  Each epoch now reports only the separate X, Y, and Z errors.  The accompanying log appended to `angle.o4249548` documents this streamlined output.

## What Removing the Wrong Metric Did

Eliminating the combined metric prevents a high error in one direction from being masked by lower errors in the others.  The per-axis results printed during training match those computed in the notebook, making it straightforward to compare folds and spot anomalies.  This change also simplifies the TensorBoard logs, which now contain three clear traces instead of an extra averaged line.

## Reproducing the Metric Cleanup

Pull the repository at this commit and run `batch_5fold_angleModel.py` or `batch_5fold_momentModel.py`.  Watch the console output to verify that exactly three metrics are reported for every epoch.  When training finishes inspect `angle.o4249548`—it should match the version committed here, confirming that the aggregate metric is gone.

## Summary of the Ninth Commit

Work then shifted toward adopting *normalized* RMSE metrics that express error as a percentage of the target range.  To gauge the effect of this change the developers executed a full training run of the angle model and saved the verbose output to a new log named `angle.o4249690`.  The resulting file spans over six thousand lines and documents every epoch of the 1000‑epoch job, including per‑axis values labeled `X_Axis_RMSE`, `Y_Axis_RMSE`, and `Z_Axis_RMSE`.  A matching placeholder file, `angle.po4249690`, was produced automatically by the cluster at job submission time.

At this stage the Python scripts themselves still computed RMSE in absolute units.  The lengthy log therefore served as a baseline against which the upcoming metric modifications could be compared.  Reviewing the console messages shows the learning rate, batch size, and validation losses used throughout the run, providing a detailed snapshot of the model’s behavior before nRMSE was fully integrated.

## What the nRMSE Trial Did

By capturing the entire training trace with the new log names, the team ensured they could later verify that introducing nRMSE did not inadvertently change the optimization dynamics.  The placeholder `.po` file highlights the use of Sun Grid Engine on the BU SCC, which creates empty prolog files when jobs are queued.  Together these artifacts mark the start of a new evaluation methodology focused on percentage‑scaled errors.

## Reproducing the nRMSE Trial

Checkout the repository at this commit and run `batch_5fold_angleModel.py` with the original environment from the early commits.  When training concludes you should see `angle.o4249690` appear in your working directory along with a zero‑byte `angle.po4249690`.  Inspect the log to confirm that three RMSE values are printed for each epoch.  These numbers are still in the original units, but the file establishes a baseline for the upcoming metric transformation.

## Summary of the Tenth Commit

The next commit completed the transition to normalized metrics.  `batch_5fold_angleModel.py` was updated with new helper functions—`X_Axis_RMSE_pct`, `Y_Axis_RMSE_pct`, and `Z_Axis_RMSE_pct`—that divide the standard RMSE by the dynamic range of each axis and multiply by 100.  The model is now compiled with these percentage‑based metrics so that every training epoch reports errors relative to the full scale of motion.  Additional lines were appended to `angle.o4249690` showing the new metric names in the training output.

## What Completing nRMSE Integration Did

Switching to nRMSE makes it possible to directly compare performance across different joints and datasets whose angle magnitudes vary widely.  The log demonstrates that the per‑axis percentages hover around the same values as the earlier absolute errors, indicating the scaling logic works correctly.  With this commit the angle model’s reporting mirrors the moment model, paving the way for consistent evaluation across tasks.

## Reproducing the Completed nRMSE Run

After pulling this revision, rerun `batch_5fold_angleModel.py`.  Watch the console or consult the updated `angle.o4249690` to verify that each epoch now prints `*_RMSE_pct` metrics.  If your output differs, double‑check that the scaler files in `SCALER/` match those expected by the script, as the percentage calculation depends on the recorded minimum and maximum values.

## Summary of the Eleventh Commit

To reduce duplication the metric functions were extracted into a small Python package called `orientpine`.  A new module `orientpine/regmetric.py` defines the full suite of RMSE and nRMSE calculations, while `orientpine/__init__.py` exposes them for import.  `batch_5fold_angleModel.py` now imports `X_Axis_RMSE_pct`, `Y_Axis_RMSE_pct`, and `Z_Axis_RMSE_pct` from this package rather than defining them inline.  Compiled bytecode files for the module were committed as well, reflecting the environment in which the scripts were run.

The accompanying log `angle.o4249690` contains yet another training run verifying that the refactored module produces identical results.  Large blocks of metric code were deleted from the main script, leaving only the data‑loading logic and model definition.  This reorganization makes it easier for other scripts and notebooks to reuse the same evaluation functions without copy‑pasting.

## What Separating Metric Functions Did

By centralizing the RMSE routines the project became much more maintainable.  Future changes to scaling or error calculation can be made in a single location and immediately apply to every training script.  The presence of the compiled `.pyc` files also hints that the developers executed the code on a cluster where the working directory was archived directly into version control.

## Reproducing the Package Refactor

Ensure that Python can locate the `orientpine` package—running the scripts from the repository root suffices.  Execute `batch_5fold_angleModel.py` and confirm that it imports the metric functions without error.  A fresh log appended to `angle.o4249690` should match the repository’s version, demonstrating that the new module behaves exactly like the earlier inline definitions.


## Summary of the Twelfth Commit

The twelfth commit did not modify any Python code but instead extended the training log `angle.o4249690`.  The researchers allowed the angle model job to continue running on the BU SCC cluster so that the log captured hundreds of additional epochs.  These lines record the evolving loss curves and per-axis errors in meticulous detail, offering a complete picture of how the network converged over time.

## What the Log Update Did

By appending the full console output, the team ensured that future comparisons could reference every epoch, not just an early subset.  The extended log reveals exactly when learning plateaued and how validation metrics fluctuated near the end of training.  Having this historical record proved valuable when tweaking hyperparameters in later runs.

## Reproducing the Log Update

Simply rerun `batch_5fold_angleModel.py` using the configuration from the previous commit.  Allow the process to reach all scheduled epochs so that `angle.o4249690` grows accordingly.  When finished, the file should contain the same expanded number of lines as the repository version.

## Summary of the Thirteenth Commit

The thirteenth commit marked a brief reversal of the earlier package refactor.  Compiled `.pyc` files and the separate `orientpine` module were removed after issues arose importing the package on the cluster.  To keep experiments moving, all metric functions were copied back into `batch_5fold_angleModel.py` itself.  A few more log entries were appended to `angle.o4249690` documenting another training run with this standalone script.

## What Folding the Metrics Back In Did

Consolidating the metrics in one file ensured the job script ran without dependency problems on the remote compute nodes.  Although less modular, this approach guaranteed reproducibility since every required function traveled with the main program.  The deletion of stale bytecode files also eliminated confusing warnings about mismatched Python versions.

## Reproducing the Consolidated Script

Checkout the repository at this commit and verify that `orientpine` no longer exists.  Execute `batch_5fold_angleModel.py` directly; it should define all RMSE and nRMSE helpers internally.  When training completes you will obtain a fresh `angle.o4249690` identical to the one committed, confirming that the script works without external modules.

## Summary of the Fourteenth Commit

With logging stabilized, the fourteenth commit prepared the pipeline for systematic model saving.  `batch_5fold_angleModel.py` gained two new variables—`SaveModelDir` and `modelVersion`—which determine where each fold stores its trained weights.  At the end of the training loop, `model.save()` writes an HDF5 file named after the goal, fold index, and version string.  This ensures that results from different experiments do not overwrite one another.

## What Preparing for Saved Models Did

Saving the models after every fold allows researchers to reload them later for evaluation or fine-tuning without repeating the entire training process.  The explicit directory and filename scheme keeps the growing collection of weights organized by experiment, which is crucial when dozens of runs are performed on shared hardware.

## Reproducing the Saved Model Workflow

Create a directory called `SavedModel` in your working tree and pull this commit.  Running `batch_5fold_angleModel.py` will now deposit files such as `angle_0_Dense_1st.h5` after each fold finishes.  Verify that these HDF5 files appear in the specified folder and that the training log still resembles `angle.o4249690`.  You can then load the saved weights in a separate script or notebook to reproduce the authors' analysis.

## Summary of the Fifteenth Commit

The fifteenth commit streamlined the repository's `.gitignore` to keep bulky cluster logs out of version control. It continued to ignore local data directories but added a short comment clarifying that qsub output files should remain untracked. Two new patterns, `angle.*` and `moment.*`, match the BU SCC's prolog and error logs which can easily reach hundreds of kilobytes each. A trailing newline fix ensures Git interprets the file correctly.

## What the Gitignore Update Did

Excluding these automatically generated logs prevents accidental commits of multi-megabyte text files every time a training job runs. It also keeps the `SavedModel` directory private so that only intentionally uploaded weights appear in history. This small housekeeping step makes subsequent diffs much easier to review.

## Reproducing the Clean Log Setup

Inspect `.gitignore` after checking out this commit. It should list the dataset and log directories followed by the comment `# qlog 파일 제외` and the two wildcard patterns. Launch `angleModel.sh` or `momentModel.sh` and verify that the resulting `angle.*` or `moment.*` files do not appear in `git status`.

## Summary of the Sixteenth Commit

After validating the metrics, the team reorganized the repository to separate training code from data preparation. All training scripts, notebooks, and job files moved into a new `modeltraining/` folder. Large log files from earlier experiments were deleted to shrink the repository, and the `.gitignore` was rewritten so only subdirectories under `modeltraining` are ignored (`modeltraining/DATASET/`, `modeltraining/SCALER/`, and so on). The renamed job scripts—`ss_angleModel.sh` and `ss_momentModel.sh`—reflect their streamlined role in launching single experiments.

## What Consolidating the Training Code Did

Grouping everything under `modeltraining` clarifies which files are needed to reproduce the main experiments. Removing thousands of lines of old logs reduces clutter and improves repository performance. The updated ignore rules ensure that future outputs stay contained within the new directory structure.

## Reproducing the Restructured Workflow

Place your dataset and scaler objects inside `modeltraining/DATASET/` and `modeltraining/SCALER/`. From the repository root run `modeltraining/ss_angleModel.sh` or `modeltraining/ss_momentModel.sh`. The scripts will save models and logs within the `modeltraining` folder hierarchy, just as they did before the move.

## Summary of the Seventeenth Commit

The seventeenth commit introduced a comprehensive data-preparation pipeline. A parallel directory named `modelpreperation/` now holds over a dozen Jupyter notebooks and helper scripts that walk through sorting raw IMU files, filtering trials, normalizing timestamps, and exporting training datasets. Example outputs in CSV and Excel format show scaled feature matrices produced with both min–max and standard scalers. An environment file `buIMU.yml` documents the Python packages required to execute these steps.

## What Adding the Preparation Notebooks Did

By publishing the entire preprocessing workflow, the authors enabled others to regenerate every dataset used in the study. The notebooks cover tasks from initial quality checks to assembling the final NumPy arrays consumed by the training scripts. The included result files serve as reference outputs so users can verify their own runs.

## Reproducing the Dataset Generation

Create a Conda environment using `conda env create -f modelpreperation/buIMU.yml`. Then open each numbered notebook in order, starting with `0_Data_sorter.ipynb` and ending with the dataset-specific scripts in step five. The generated CSV and Excel files should match those under `modelpreperation/result_processingData/`. Once complete, copy the resulting `.npz` archives and scaler pickles into `modeltraining/DATASET/` and `modeltraining/SCALER/` to train models exactly as the original authors did.

## Summary of the Eighteenth Commit

To keep the growing preparation notebooks organized, the eighteenth commit created a dedicated `StudyRoom/` subdirectory.  Two study materials—`Study_checkpoint.ipynb` and a sample CSV file named `afterPDFCHKforSenddance.csv`—were relocated from the root of `modelpreperation/` into this new folder.  No code changed, but the new path clarifies that these files document exploratory analyses rather than steps in the main dataset pipeline.

## What Reorganizing the Study Files Did

Placing the supplemental notebook and CSV under `modelpreperation/StudyRoom/` separates reference material from the scripts that actually generate training data.  This small move helps future contributors quickly locate background experiments without confusing them with the core workflow.

## Reproducing the Study Setup

After pulling this commit, look under `modelpreperation/StudyRoom/` to find `Study_checkpoint.ipynb`.  Open the notebook to inspect the exploratory plots or rerun the cells if desired.  The associated CSV remains in the same folder so the notebook can load it with a relative path.

## Summary of the Nineteenth Commit

The nineteenth commit performed an extensive cleanup and renaming pass.  The `modelpreperation/` directory was renamed to `preperation/` while `modeltraining/` became simply `training/`.  All notebooks, scripts, and result files moved accordingly.  In the process, the team committed a full `training/DATASET/IWALQQ_1st` folder containing the finalized five-fold split and matching scaler objects.  TensorBoard logs and the main training scripts were also migrated into the `training/` hierarchy.

## What the Code Cleanup Changed

Renaming the top-level folders standardized the project layout and made it obvious which files prepare data versus train models.  Including the ready-to-use dataset allows anyone to run the training scripts immediately without regenerating the `.npz` archives.  The added log directories show exactly how TensorFlow was configured during these reference runs.

## Reproducing the Unified Layout

Check that your working tree now contains `preperation/` and `training/`.  The dataset for the IWALQQ_1st experiment lives under `training/DATASET/`.  Execute `training/ss_angleModel.sh` or `training/ss_momentModel.sh` to train using these files; TensorBoard traces will appear under `training/logs/fit/` matching the repository’s structure.

## Summary of the Twentieth Commit

A short follow-up commit adjusted the output location for saved models on the BU SCC cluster.  Both `training/angle_Model.py` and `training/moment_Model.py` now write their HDF5 files to `/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel/` instead of a more generic folder.

## What Updating the Save Path Did

By storing models under a project-specific directory, the researchers ensured that multiple experiments would not overwrite one another on shared storage.  This change also simplifies archiving trained weights alongside the source code.

## Reproducing the Updated Save Path

Open either training script and verify that the `SaveDir` variable points to the new location.  When you run the scripts on the SCC cluster, the generated model files should appear under the specified path, confirming that the change takes effect.


## Summary of the Twenty-First Commit

The twenty-first commit introduced a brand-new dataset variant named `IWALQQ_2nd`.  The developers revised `preperation/4_DataSet_CAN_MYWAY.ipynb` so that it generated this second round of input files.  Numerous code cells were updated to load a different collection of raw recordings, apply the same normalization steps as before, and then save the results under distinct filenames.  By creating `IWALQQ_2nd` the team could test their models on an alternative subject split without rerunning the entire preprocessing pipeline from scratch.

## What the New Dataset Added

Having two independent datasets allowed the researchers to verify that their models generalized across separate recording sessions.  The notebook changes ensured the second dataset mirrored the feature ordering and scaling of the first, making it easy to swap between them in the training scripts.  Future commits rely on these files when exploring more complex architectures.

## Reproducing the Dataset Expansion

Open `preperation/4_DataSet_CAN_MYWAY.ipynb` and execute the cells after cloning this commit.  The notebook will output `.npz` archives and scaler pickles labeled for `IWALQQ_2nd`.  Move these files into `training/DATASET/` and `training/SCALER/` so the existing scripts can use them during training.

## Summary of the Twenty-Second Commit

This commit was a merge from the remote `main` branch that synchronized a large set of artifacts.  Besides incorporating the freshly created dataset, it pulled in TensorBoard logs, updated notebooks, and the finalized `training/DATASET/IWALQQ_1st` directory complete with all five folds.  A handful of job scripts and Python files were also adjusted to reflect the latest directory names.

## What Merging the Remote Changes Did

Combining both histories ensured every collaborator had the same starting point for upcoming experiments.  The imported logs document previous runs in detail, while the bundled dataset makes it trivial to rerun baseline models.  Although the merge touched hundreds of files—including binary profiles—the functional code remained unchanged, serving mostly to align local and remote development.

## Reproducing the Merge Results

After checking out this commit you should see the expanded `training/` folder with its dataset subdirectories and TensorBoard log trees.  Running any of the training scripts will now produce outputs consistent with those logs.  No additional setup is required beyond verifying that the new files exist.

## Summary of the Twenty-Third Commit

Once the repository was in sync, the twenty-third commit cleaned up the working tree for fresh experiments.  The `.gitignore` rules were rewritten to point at the renamed `training/` directories, preventing future logs and datasets from being accidentally committed.  A new notebook, `training/StudyRoom/load_modelNvisualize.ipynb`, demonstrates how to load saved HDF5 models and plot their predictions against ground truth.  At the same time, obsolete TensorBoard profile files were truncated to zero bytes, dramatically reducing repository size.  Minor edits to both training scripts switched their model construction to use the simpler `tf.keras.Sequential` API.

## What Preparing for Training Did

By tidying the log directories and adding the visualization notebook, this commit set the stage for systematic experimentation.  The notebook offers a template for evaluating trained networks without rerunning the full training loop, while the slimmer `.gitignore` keeps only the necessary artifacts under version control.  Updating the model code ensures compatibility with later versions of TensorFlow.

## Reproducing the Preparation Stage

Clone the repository at this revision and confirm that running `training/angle_Model.py` or `training/moment_Model.py` generates logs excluded by the new `.gitignore`.  Then open `training/StudyRoom/load_modelNvisualize.ipynb` to practice loading a saved model and generating the diagnostic plots.  The repository should remain lightweight because the large profiling files are now empty placeholders.


## Summary of the Twenty-Fourth Commit

The twenty-fourth commit made a tiny adjustment to the cluster job script `training/ss_momentModel.sh`.  A stray blank line following the memory request options was deleted so the resource directives appear together.  While functionally identical, this tidy formatting reflects the authors' habit of keeping the submission files neatly organized before queueing new jobs on the BU SCC.

## What the Alignment Cleanup Did

Removing the extra line helps ensure that future edits to the job script are obvious in diff views.  It also mirrors the layout of the matching angle-model script, so collaborators can quickly compare the two when preparing submissions.

## Reproducing the Shell Script Update

Open `training/ss_momentModel.sh` after pulling this commit and verify that the module-loading command immediately follows the memory settings with no blank lines in between.  Submitting the script to the queue should behave exactly as before.

## Summary of the Twenty-Fifth Commit

Just before launching another round of experiments, the team performed a final sweep through `training/angle_Model.py` and `training/moment_Model.py`.  Debug print statements reporting the TensorFlow and Keras versions were removed, leaving a clean banner when the scripts start.  More importantly, the path for TensorBoard logs was rewritten using `os.path.join` so that logs are grouped under `logs/fit/<modelVersion>/<nameDataset>/` followed by a timestamp and fold number.  This hierarchical layout keeps results from different models and datasets separate.

## What the Pre-Submission Edits Did

These tweaks minimize clutter in the console output and make it trivial to compare runs.  By nesting log directories under the model and dataset names, TensorBoard dashboards stay organized even when dozens of experiments are queued on the cluster.

## Reproducing the Final Training Setup

After checking out this commit, run either training script.  Examine the newly created log directory structure under `training/logs/fit/` and confirm that version and dataset subfolders appear.  The absence of `print(tf.__version__)` lines in the console output confirms that the cleanup took effect.

## Summary of the Twenty-Sixth Commit

The twenty-sixth commit represents the very last tweaks made before submitting the initial experiments.  Both training scripts dropped an unnecessary `Flatten` layer because the input vectors were already one-dimensional.  The notebook `training/StudyRoom/load_modelNvisualize.ipynb` was executed from start to finish, capturing example output such as TensorFlow warnings and model summaries within the saved JSON.  This provides a reference for how to load a trained model and visualize its predictions.

## What the Last-Minute Notebook Tweaks Did

Executing the notebook ensures that all code cells run without error and that the displayed model architecture matches the streamlined scripts.  Removing the redundant `Flatten` layer slightly reduces parameter count and clarifies the intended input shape.

## Reproducing the Submission Revision

Run `training/angle_Model.py` or `training/moment_Model.py` to train models with the simplified architecture.  Open the updated notebook and step through it to confirm that models load correctly and produce plots of predicted versus true values.  The outputs recorded in the repository should mirror what you see locally, verifying that the final configuration matches the authors' submission state.


## Summary of the Twenty-Seventh Commit

In this commit the authors locked in the final hyperparameter settings before submitting long training jobs to the BU SCC. The notebooks and scripts from the prior revision were committed again after verifying that the model version should be labelled `Dense_1st` and that the `IWALQQ_1st` dataset was the target. Functionally the code matches the previous commit, but re-saving the notebook ensured that all cells executed cleanly on the cluster. Both `angle_Model.py` and `moment_Model.py` continued to omit the unused `Flatten` layer, and the example notebook output was updated to reflect the finalized configuration.

## What Finalizing the Training Setup Did

By reiterating the parameters and notebook execution, the team documented the exact state of the code that would be queued on the cluster for extensive runs. This snapshot provides future researchers with a precise reference should they wish to replicate the original submission environment.

## Reproducing the Confirmed Configuration

Checkout this commit and open `training/StudyRoom/load_modelNvisualize.ipynb` to confirm that all cells run without modification. Next, execute either training script and verify that the console shows the `Dense_1st` model version and the `IWALQQ_1st` dataset name. The resulting logs should appear under `training/logs/fit/Dense_1st/IWALQQ_1st/` matching those archived in the repository.

## Summary of the Twenty-Eighth Commit

The twenty-eighth commit is a straightforward merge of changes from the remote repository. No files were altered beyond the automatic merge metadata. This step synchronized the local history with collaborators before launching additional experiments.

## What the Merge Accomplished

Merging ensured that any tweaks pushed by teammates—such as minor notebook edits or log updates—were incorporated without manual cherry-picking. Although no code changed, recording the merge preserves the full development timeline.

## Reproducing the Merge State

There is nothing to execute for this commit. Simply note that the repository history now contains a merge node linking the parallel lines of development.

## Summary of the Twenty-Ninth Commit

Immediately after syncing the repository, the authors kicked off training on a new dataset variant. Both `training/angle_Model.py` and `training/moment_Model.py` were edited so the `nameDataset` variable reads `IWALQQ_2nd`. Everything else—including the model architecture, optimizer settings, and logging scheme—remained untouched.

## What Switching Datasets Did

Pointing the scripts at `IWALQQ_2nd` allowed the researchers to evaluate how models trained on the second data split performed compared to the original `IWALQQ_1st` runs. Because all other settings stayed the same, any differences in the resulting metrics can be attributed to the change in training data.

## Reproducing the IWALQQ_2nd Experiments

Pull this commit and run the two training scripts. New log folders will appear under `training/logs/fit/Dense_1st/IWALQQ_2nd/`. Compare these outputs against the earlier `IWALQQ_1st` results to gauge the effect of the alternate dataset.

## Summary of the Thirtieth Commit

Commit thirty extended the visualization notebook `training/StudyRoom/load_modelNvisualize.ipynb` so that results from the `IWALQQ_1st` training runs could be inspected in detail. New cells parse the log directories, compute per-fold metrics, and draw plots comparing predicted and true joint angles across the gait cycle.

## What the Additional Plots Showed

These enhancements made it easy to spot trends across the five folds and verify that training converged as expected. The added graphs also served as reference figures for later reports.

## Reproducing the Visualization

Run the updated notebook after a set of `IWALQQ_1st` experiments completes. The new cells will generate summary plots and tables identical to those archived in the repository.

## Summary of the Thirty-First Commit

The thirty-first commit corrected several typographical errors in the dataset-preparation notebooks and refined the angle error metric. File names referencing the moment variable were standardized to `moBWHT`, and the metric calculation was updated in both the notebooks and `training/angle_Model.py`.

## What the Typo and Metric Fixes Did

Cleaning up these mistakes ensured that all scripts used consistent variable names, preventing subtle bugs during data loading. The metric revision provided more accurate feedback during training and visualization.

## Reproducing the Fixed Workflow

Rerun the data-preparation notebooks to regenerate the cleaned datasets, then execute the training scripts. Logs should now reference `moBWHT` and report the corrected angle metric.

## Summary of the Thirty-Second Commit

In this commit the authors regenerated the datasets with a consistent random seed and added `preperation/list_dataset.xlsx` to track each split. To keep the repository lightweight, the large `.npz` and scaler files under `training/DATASET/IWALQQ_1st/` were replaced with empty placeholders. The `.gitignore` rules were expanded so future datasets and normalization outputs would not be committed.

## What Regenerating the Data Achieved

Standardizing on a single seed guarantees that collaborators can reproduce the exact same splits while also documenting the dataset catalog. Removing the heavyweight data files keeps the repository manageable without losing the directory structure.

## Reproducing the Data Generation

Follow `preperation/4_DataSet_CAN_MYWAY.ipynb` with the specified seed to create fresh training and test sets. Store them outside the repository as directed by `.gitignore` and verify that the Excel file lists each generated fold.


## Summary of the Thirty-Third Commit

The thirty-third commit aligned both training scripts with the finalized data directory structure. Instead of loading datasets from the local `training/DATASET/` folder, `angle_Model.py` and `moment_Model.py` now point to `../preperation/SAVE_dataSet/IWALQQ_1st`. A new variable named `relativeDir` stores this path so it can be easily adjusted in the future. The moment model script previously referenced the `IWALQQ_2nd` split; this commit changes it to `IWALQQ_1st` so both angle and moment predictions are trained from the same five-fold dataset.

## What Unifying the Data Paths Did

By referencing the shared preparation directory, the scripts avoid duplicating large datasets under `training/`. This ensures that any updates to the saved data—such as regenerated folds or new scaler objects—will automatically be used the next time the training scripts run. Using the same dataset name across both models also eliminates the risk of comparing results from mismatched splits.

## Reproducing the Updated Training Scripts

Place the `IWALQQ_1st` dataset under `preperation/SAVE_dataSet/` as expected by the new `relativeDir` variable. Then run `training/angle_Model.py` or `training/moment_Model.py`. Each script will load the data from this location and write logs under `training/logs/fit/Dense_1st/IWALQQ_1st/`, confirming that the path adjustment works.

## Summary of the Thirty-Fourth Commit

Before launching long training jobs, the authors reran the metrics notebook one last time. `training/StudyRoom/Check_metric.ipynb` was updated to read from the new `preperation/SAVE_dataSet/IWALQQ_1st` folder and to use the corrected moment variable name `moBWHT`. The notebook was executed from scratch so the saved JSON contains fresh outputs showing TensorFlow initialization messages, array shapes, and example error calculations.

## What the Final Metric Check Did

Running the notebook with the finalized paths verified that the scaler files and dataset contents matched the expectations of the training scripts. It also confirmed that the `moBWHT` naming convention was consistent throughout the codebase. The captured outputs demonstrate that RMSE calculations run without issue on the prepared data.

## Reproducing the Metric Validation

Open `training/StudyRoom/Check_metric.ipynb` and execute each cell. The notebook should load fold zero of the `IWALQQ_1st` dataset, apply the scalers, and print the same tensor shapes shown in the repository version. Examine the final cells to ensure the rescaled RMSE numbers align with those in the saved output.

## Summary of the Thirty-Fifth Commit

This follow-up commit corrected lingering typos in `training/moment_Model.py`. Every reference to `moBHWT` was replaced with `moBWHT`, affecting the scaler loading commands, metric functions, and debugging print statements. No functional code changed beyond these string updates, but the corrections keep the terminology uniform across scripts and notebooks.

## What Fixing the Typos Did

Consistent variable names prevent file-not-found errors when the script attempts to load scaler pickles and ensure that metrics operate on the intended arrays. With the typos removed, the moment model can train and evaluate without manual edits, and logs clearly match the dataset files on disk.

## Reproducing the Cleaned-Up Training Script

After pulling this commit, run `training/moment_Model.py` with the `IWALQQ_1st` dataset in place. The script should locate `*_moBWHT.pkl` scaler files, compile the network, and begin training while reporting X‑, Y‑, and Z‑axis RMSE percentages. The absence of missing-file errors confirms that the typo fix was successful.

## Summary of the Thirty-Sixth Commit

The thirty-sixth commit compared two preprocessing workflows for generating training data. `preperation/4_DataSet_CAN_MYWAY.ipynb` and `preperation/4_DataSet_NAC_Mundt.ipynb` were updated and executed from scratch. Their outputs include numerous CSV files under `preperation/mundtway_minmaxscaler_result/` that contain IMU signals, joint angles, and moment data both before and after min–max scaling. An older `column_wise_scaledResult.csv` was removed in favor of these new per-method results, and several intermediate Excel sheets were added so that the exact scaling steps could be reviewed.

## What Comparing Preprocessing Methods Did

By saving the scaled datasets for both the authors’ custom approach and the protocol described by Mundt et al., the team could directly inspect how each technique normalized the raw signals. The commit makes it possible to reproduce plots or metrics using either method, enabling an informed decision about which preprocessing pipeline yields better model accuracy.

## Reproducing the Preprocessing Comparison

Run `preperation/4_DataSet_CAN_MYWAY.ipynb` and `preperation/4_DataSet_NAC_Mundt.ipynb` in order. Each notebook reads the raw IMU exports, applies its respective scaling strategy, and writes CSV files to `preperation/mundtway_minmaxscaler_result/`. Comparing these outputs will reveal the differences between the two preprocessing schemes.

## Summary of the Thirty-Seventh Commit

Preparing for an upcoming meeting, the authors refined `training/StudyRoom/load_modelNvisualize.ipynb`. Comment blocks now emphasize verifying which scaler is loaded when computing metrics, and lingering `moBHWT` typos were corrected to `moBWHT`. The notebook was rerun from a clean state so its execution counts and plotted outputs serve as a reference for future demonstrations.

## What the Visualization Updates Did

These changes ensure that anyone reviewing the notebook understands the dependency on the correct scalers and sees consistent results when reloading trained models. The clearer annotations help collaborators follow the metric calculations step by step.

## Reproducing the Updated Visualization

Open `training/StudyRoom/load_modelNvisualize.ipynb` and run all cells. The notebook will load the latest saved models, apply the `moBWHT` scalers, and display the angle and moment trajectories. Verify that your figures match those embedded in the repository.

## Summary of the Thirty-Eighth Commit

Two job scripts—`training/ss_angleModel.sh` and `training/ss_momentModel.sh`—were modified to request a 24‑hour wall time instead of 48 hours. No other scheduling parameters changed.

## What Shorter Wall Times Did

Reducing the requested runtime prevents jobs from monopolizing resources on the BU SCC cluster and fits them within typical daily queue limits. The training code itself is unchanged but will terminate earlier if it does not finish within one day.

## Reproducing the Adjusted Job Scripts

Submit either script to the cluster using `qsub`. The scheduler will enforce the 24‑hour limit, after which your job will stop and save logs in `result_qsub/`.

## Summary of the Thirty-Ninth Commit

After confirming the shorter wall times, the repository was synchronized with the latest changes from the remote GitHub repository.  This merge incorporated small tweaks that other collaborators had made, most notably adjustments to the visualization notebook used for inspecting trained models.  The commit message simply notes “Merge branch 'main' of https://github.com/orientpine/IMUforKnee,” but the diff shows that `training/StudyRoom/load_modelNvisualize.ipynb` gained additional plotting cells while redundant code was pruned.

## What the Merge Brought In

Merging ensured everyone was operating from the same history before more ambitious experiments began.  The updated notebook clarifies how to load specific scaler files and includes example figures generated on the cluster.  No Python scripts were modified, but having the refined notebook in version control allowed team members to verify results without repeating the entire analysis.

## Reproducing the Merge State

Checkout this commit and open `training/StudyRoom/load_modelNvisualize.ipynb`.  You should see the extra plots and comments that appeared in the merge.  Running the notebook with your own saved models will produce figures identical to those stored in the repository, confirming that the merge did not introduce conflicts.

## Summary of the Fortieth Commit

With the codebase unified, attention turned to reducing resource usage on the BU SCC.  Both training scripts switched back to the `IWALQQ_2nd` dataset, and new “lean” job scripts were created to request only four CPU cores and one GPU.  The existing `ss_angleModel.sh` and `ss_momentModel.sh` scripts were updated accordingly and now refer readers to BU documentation for memory guidelines.  These adjustments keep queue wait times reasonable while still allowing full training runs.

## What the Resource Reduction Did

By trimming the parallelism and memory requests, the team could launch more experiments simultaneously without overloading the cluster.  The lean versions of the job scripts serve as templates for quick tests that need less than a day to complete.  Switching datasets ensured the changes were evaluated on a fresh split, providing another point of comparison for model generalization.

## Reproducing the Resource Change

Use `training/ss_lean_angleModel.sh` or `training/ss_lean_momentModel.sh` to submit lightweight jobs.  Each script activates the same `sccIMU` environment but allocates only four OpenMP threads.  Logs will appear under `result_qsub/angle_lean` or `result_qsub/moment_lean`.  Verify that your run targets the `IWALQQ_2nd` dataset as specified in the Python scripts.

## Summary of the Forty-First Commit

This commit marks the project’s first exploration of PyTorch.  New example files—`tmp_torch.py`, a minimal convolutional network, and `tmp_torch_angle_Model.ipynb`, a companion notebook—demonstrate how to build and run models using PyTorch tensors.  A helper script `ss_tmp.sh` launches these prototypes on the cluster with a short ten‑minute time limit.  The standard job scripts were adjusted to use the `omp` parallel environment, ensuring consistent thread handling across TensorFlow and PyTorch runs.

## What Starting the PyTorch Port Did

Introducing PyTorch opened the door for more flexible model architectures and easier GPU management.  The toy network shows how to move computations to CUDA when available and serves as a foundation for future experiments.  By committing both a script and a Jupyter notebook, the authors provided templates for batch jobs and interactive debugging alike.

## Reproducing the PyTorch Experiment

Activate the `torchIMU` environment referenced in `ss_tmp.sh` and submit the script to your cluster.  Alternatively, run `tmp_torch_angle_Model.ipynb` locally to step through the example network.  You should see a simple convolutional model instantiated and the chosen device (CPU or GPU) printed to the console, matching the behavior captured in the repository’s notebook.

## Summary of the Forty-Second Commit

To verify that PyTorch could utilize the available GPUs, the team expanded the temporary scripts introduced earlier.  `training/tmp_torch.py` now imports `torch` and `torchvision` and prints the detected device.  The companion notebook `tmp_torch_angle_Model.ipynb` gained a new markdown header and code cells that call `torch.cuda.is_available()` before instantiating the toy convolutional network on CUDA.  A simple text file named `training/tmp` records the word “cuda” to indicate the environment where the test succeeded.

## What Confirming GPU Execution Did

Running the updated notebook proved that the cluster configuration exposed NVIDIA GPUs to PyTorch just as it did for TensorFlow.  Seeing the device string "cuda" reassured the researchers that future models could train on GPU without additional setup.

## Reproducing the GPU Test

Activate the `torchIMU` environment and execute `tmp_torch_angle_Model.ipynb`.  The notebook should report that CUDA is available and print the GPU name when constructing the model.  Alternatively, run `python training/tmp_torch.py` from the command line and check that the same message appears.

## Summary of the Forty-Third Commit

With GPU access confirmed, work began on porting the existing Keras models to PyTorch.  Early in this process the developers experimented with the built‑in `FashionMNIST` dataset as a stand‑in for the IMU data.  Several binary files under `training/data/FashionMNIST/FashionMNIST/raw/` were added so the notebook could download the dataset once and reuse it.  `tmp_torch_angle_Model.ipynb` was reworked to load these images and train a simple classifier while mirroring the structure of the TensorFlow code.

## What the Migration Prototype Did

This commit served as a learning exercise to replicate the Keras workflow using PyTorch’s `DataLoader` and module system.  Although the FashionMNIST images are unrelated to joint angle estimation, they provided a lightweight dataset for debugging the training loop.  The notebook records the model’s accuracy and illustrates how to save checkpoints in the new framework.

## Reproducing the Migration Prototype

Run the notebook from start to finish.  If the FashionMNIST files are absent, PyTorch will download them automatically.  Training for a few epochs should produce accuracy numbers similar to those shown in the commit.  Inspect the saved model file and compare it with the Keras version to see how the formats differ.

## Summary of the Forty-Fourth Commit

After testing the FashionMNIST example, the bulky dataset files were removed from version control to keep the repository small.  All eight binary archives in `training/data/FashionMNIST/FashionMNIST/raw/` were deleted.  The notebook remains intact but now expects the data to be fetched on demand.

## What the Cleanup Achieved

By stripping out the 60+ MB of sample images, the commit restored a manageable repository size while still allowing the PyTorch example to run.  Future clones will download the dataset automatically, preventing unnecessary bloat.

## Reproducing the Cleanup State

Pull this commit and execute the FashionMNIST notebook again.  The first cell will trigger a download if the raw files are missing, recreating the environment used for the PyTorch tests.

## Summary of the Forty-Fifth Commit

Attention then returned to data preparation.  Two notebooks—`preperation/0_Data_sorter.ipynb` and `preperation/1_Data_Checker.ipynb`—were revised to correct axis orientation issues discovered in earlier datasets.  New markdown cells describe how to append the subject side information when converting text files to CSV, and code cells were updated with the corrected directory paths.  Several execution counts were reset, indicating the notebooks were rerun from scratch after making the adjustments.

## What the Axis Correction Did

Correcting the axis order ensures that subsequent training scripts interpret the IMU signals consistently.  The updated notebooks reorganize the preprocessing pipeline so that the X, Y, and Z columns match the physical sensor axes used during data collection.  This fix is crucial for comparing results across different recording sessions.

## Reproducing the Axis Correction

Open `0_Data_sorter.ipynb` and `1_Data_Checker.ipynb` in Jupyter and run all cells.  Provide the raw data directories specified near the top of each notebook.  When they finish, the processed CSV files should contain columns in the new axis order and omit any zero‑only columns, matching the state of the repository after this commit.

## Summary of the Forty-Sixth Commit

The forty-sixth commit dramatically expanded the PyTorch prototype into a full training setup. Two preparation notebooks were updated with additional cells that append side information when converting raw text files to CSV and reorganize directories for the corrected dataset. A new `StudyRoom` folder under `training/` contains multiple speed-check notebooks and Python scripts that benchmark small models. The main Torch scripts—`torch_angleModel.py` and `torch_momentModel.py`—were introduced along with qsub wrappers `ss_torch_angleModel.sh` and `ss_torch_momentModel.sh`. Several short TensorBoard logs demonstrate that these scripts ran successfully on the SCC GPU nodes.

## What the Early PyTorch Integration Did

By creating dedicated Torch versions of the angle and moment models, the researchers began migrating away from Keras. The notebooks show how to load IMU data using `DataLoader`, compute nRMSE metrics on the GPU, and save scripted models in the new `training/model_scripted.pt` path. Temporary speed tests helped tune batch sizes and verify that dataset loading kept up with GPU throughput.

## Reproducing the Early PyTorch Run

Start by checking out this commit and installing PyTorch in a fresh environment. Run `training/StudyRoom/tmp_torch_angle_Model.ipynb` to confirm CUDA availability, then launch `ss_torch_angleModel.sh` on your cluster. TensorBoard logs should appear alongside the sample event files committed in the repository, showing that the Torch models train without errors.

## Summary of the Forty-Seventh Commit

This commit simply merged the latest changes from the remote repository. No files were modified beyond resolving version control metadata.

## What the Merge Did

The merge synchronized the new PyTorch scripts with earlier Keras work, ensuring that collaborators shared an identical history before continuing development.

## Reproducing the Merge State

After pulling this commit you should see no differences when running `git status`. The repository reflects both lines of work without conflicts.

## Summary of the Forty-Eighth Commit

Two empty placeholder files named `testis` and `testis2` were deleted from the `training` directory. They were remnants of debugging and served no purpose in the final codebase.

## What the Cleanup Did

Removing these zero-byte files tidied the repository and avoided confusion about whether they were required inputs.

## Reproducing the Cleanup State

Verify that `training/` contains no `testis` or `testis2` files. If they reappear after running earlier commits, delete them to match this state.

## Summary of the Forty-Ninth Commit

Data preparation resumed with the addition of `list_dataset_correction.xlsx` and updates to two preprocessing notebooks. The scripts now write their normalized CSV files under `NORM_CORRECTION` and generate a dataset called `IWALQQ_1st_correction`. `.gitignore` was updated so the new folder is not tracked.

## What the Dataset Addition Did

These changes produced a cleaned and relabeled dataset that corrected previous naming mistakes and ensured consistent axis orientation. The Excel file enumerates all source recordings used to build the set.

## Reproducing the New Dataset

Run `preperation/3_0_Data_filtertoSave.ipynb` followed by `preperation/4_DataSet_CAN_MYWAY.ipynb`. Provide paths to your raw CSVs and confirm that the resulting NumPy archives are saved under `preperation/NORM_CORRECTION` with names matching `IWALQQ_1st_correction`.


## Summary of the Fiftieth Commit

The fiftieth commit shifted training over to the freshly corrected dataset. Both Torch job scripts were retuned for longer runs and higher thread counts, writing their output to new `result_qsub` files. The `torch_angleModel.py` and `torch_momentModel.py` programs now load `IWALQQ_1st_correction` and record separate log directories for angle and moment data. Learning rate and batch size defaults were tightened to `0.0002` and `16`. TensorBoard writers now use the data type in their path, and nRMSE helpers call `.item()` so scalar values are stored in Python variables rather than GPU tensors.

## What Training on the Corrected Data Did

Switching to the `IWALQQ_1st_correction` dataset validated that the preprocessing fixes behaved as expected. By lowering the batch size, the scripts fit easily on a single GPU while still producing regular progress logs. Explicit conversion of nRMSE tensors prevented warnings when aggregating metrics, ensuring consistent reporting.

## Reproducing the Corrected Dataset Training

Set up PyTorch and the `IWALQQ_1st_correction` data files under `preperation/SAVE_dataSet`. Submit `training/ss_torch_angleModel.sh` and `training/ss_torch_momentModel.sh` to your cluster. TensorBoard logs should appear in `training/logs/pytorch/*/Dense_1st_torch/IWALQQ_1st_correction/` with filenames that include `angle` or `moBWHT`.

## Summary of the Fifty-First Commit

This commit further tweaked the Torch scripts to test a more aggressive configuration. The learning rate increased to `0.001` and the batch size jumped to `256`. A misspelled variable `lreaningRate` was corrected to `learningRate`, and the optimizer references were updated accordingly.

## What the Hyperparameter Trial Did

Raising the learning rate and batch size explored how quickly the Dense model could converge on the corrected data. These changes generated larger GPU loads and reduced epoch times, offering a baseline for high-throughput experiments.

## Reproducing the Hyperparameter Trial

After applying this commit, launch the same job scripts. Confirm that each epoch prints the larger batch size and that the `NAdam` optimizer uses a `0.001` learning rate. Monitor `result_qsub/torch_*` logs to track convergence speed.

## Summary of the Fifty-Second Commit

To capture configuration details alongside the metrics, the next commit wrote hyperparameter summaries directly into TensorBoard. `ss_torch_angleModel.sh` now writes to `result_qsub/torch_angle_4`, and `torch_angleModel.py` appends `add_hparams` sections for every run. These records include the learning rate, batch size, and dataset name with associated loss and nRMSE values.

## What the Hyperparameter Logging Did

Logging the parameters makes it easy to compare multiple experiments in TensorBoard’s HParams plugin. Researchers can now sort runs by dataset or learning rate and visualize their effect on training curves without digging through scripts.

## Reproducing the Hyperparameter Logging

Run `ss_torch_angleModel.sh` again. When TensorBoard opens, navigate to the HParams tab and select runs from `torch_angle_4`. The table should list the `lr`, `bsize`, and `DS` fields for each fold, with final loss and nRMSE metrics beside them.

## Summary of the Fifty-Third Commit

The fifty-third commit focused on generating comprehensive PDFs and a new CSV summarizing the checked recordings. Several preparation notebooks were rerun so their execution counts match the latest steps, and `afterPDFCHKforSenddance.csv` captures the validated file list. The `2_Data_PDFViewNCheck.py` script was expanded to automate opening each PDF, waiting for user input, and copying selected entries into the results sheet.

## What the New PDF Workflow Did

By scripting the PDF review process, the team ensured that only high-quality motion trials were included in subsequent datasets. The updated notebooks incorporate these selections, producing normalized CSVs and NumPy archives that exactly match the curated list.

## Reproducing the PDF Generation

Execute `preperation/1_Data_Checker.ipynb` through `preperation/4_DataSet_CAN_MYWAY.ipynb` in order. As you review each PDF with `2_Data_PDFViewNCheck.py`, press the indicated keys to mark good trials. Once complete, verify that `afterPDFCHKforSenddance.csv` contains your chosen files and that the notebooks generate corresponding datasets.


## Summary of the Fifty-Fourth Commit

This merge reconciled the recent PDF-review work with earlier hyperparameter logging changes. A new Excel spreadsheet `preperation/list_dataset_correction.xlsx` lists the corrected motion-trial files that make up `IWALQQ_1st_correction`. Both `3_0_Data_filtertoSave.ipynb` and `4_DataSet_CAN_MYWAY.ipynb` were rerun so their execution counts matched the merged branch. The `.gitignore` gained an entry for `preperation/NORM_CORRECTION` to keep intermediate normalization files out of version control.

The Torch job scripts grew more robust as well. `ss_torch_angleModel.sh` and `ss_torch_momentModel.sh` now request a two‑hour wall time and eight OpenMP threads while directing logs to new `result_qsub` paths. Inside `torch_angleModel.py` and `torch_momentModel.py` the dataset name changed to `'IWALQQ_1st_correction'`, the learning rate variable was spelled correctly as `learningRate`, and the TensorBoard writers include the data type in their folder names. Calls to the nRMSE helper functions end with `.item()` so that each epoch’s totals are stored as plain floats.

### What the Merge Achieved

By combining the corrected dataset, hyperparameter logging, and PDF workflow into a single state, the repository captured all recent fixes in one place. The expanded job scripts ensured longer runs would not time out, while the Torch programs logged cleaner metrics thanks to the scalar conversion. The added spreadsheet preserves exactly which recordings belong to the corrected dataset.

### Reproducing the Merge State

Checkout this commit and place your cleaned CSV files under `preperation/NORM_CORRECTION`. Execute `preperation/3_0_Data_filtertoSave.ipynb` and `4_DataSet_CAN_MYWAY.ipynb` to regenerate the archives listed in `list_dataset_correction.xlsx`. Then submit `ss_torch_angleModel.sh` or `ss_torch_momentModel.sh` and confirm that logs appear under `result_qsub/torch_angle_4` and `torch_moment_3`.

## Summary of the Fifty-Fifth Commit

This short update captured the results of a new moment‑model training run. The only change moved `ss_torch_momentModel.sh`’s output directory from `result_qsub/torch_moment_3` to `result_qsub/torch_moment_4`, separating the fresh logs from earlier tests.

### What the New Results Provided

By directing output to a new folder the researchers kept each experiment’s logs organized. This made it easier to compare runs without overwriting previous results.

### Reproducing the Moment Run

Submit `ss_torch_momentModel.sh` after applying this commit. You should find a new file in `result_qsub/torch_moment_4` once the job completes.

## Summary of the Fifty-Sixth Commit

With the next experiment the team reduced the learning rate to `0.0001` and dropped the batch size back to `16` for both angle and moment models. The job scripts now write to `result_qsub/torch_angle_5` and `torch_moment_5`. In the Python trainers, the hyperparameter dictionaries passed to `add_hparams` include `sess` and `Type` fields so TensorBoard can distinguish training versus testing runs and whether they model angles or moments.

### What the Low‑LR Trial Explored

This configuration tested whether a smaller learning rate would stabilize training on the corrected dataset. The revised hyperparameter logging captures each fold’s settings along with final loss and nRMSE values, enabling side‑by‑side comparisons in TensorBoard’s HParams tab.

### Reproducing the Low‑LR Trial

Run both job scripts after checking out this commit. When TensorBoard reads `training/logs`, select entries from `torch_angle_5` or `torch_moment_5` and examine their hyperparameter tables to confirm the `lr=0.0001` and `bsize=16` settings.

## Summary of the Fifty-Seventh Commit

Here the loss function switched from mean absolute error to a custom root‑mean‑square error. `torch_angleModel.py` and `torch_momentModel.py` define an `RMSELoss` class wrapping PyTorch’s `MSELoss`; both scripts now instantiate this class and record the loss function name alongside other hyperparameters. Job logs are routed to `result_qsub/torch_angle_6` and `torch_moment_6`, and the visualization notebook under `training/StudyRoom` was reset so its execution count starts at one.

### What Changing the Loss Function Did

Using RMSE emphasizes larger errors and can yield smoother convergence when predicting continuous joint trajectories. The updated hyperparameter logging notes `lossFunc="RMSE"`, ensuring analysts can trace which metric produced each result. New log folders prevent confusion with earlier MAE‑based experiments.

### Reproducing the RMSE Experiment

Launch the updated job scripts and open TensorBoard to verify that the run directories include `lossFunc: RMSE` in their hyperparameters. Inspect the resulting curves to judge whether RMSE improves stability compared to the previous MAE runs.

## Summary of the Fifty-Eighth Commit

To make future comparisons easier, the training scripts were restructured so the
choice of loss function can be set with a single variable.  Both
`torch_angleModel.py` and `torch_momentModel.py` now define a small
`makelossFuncion` helper that returns either an `RMSELoss` or `nn.L1Loss`
depending on the `lossFunction` string.  Default hyperparameters still use
root‑mean‑square error, but switching to MAE or other criteria only requires
changing one line.  The new variable is also logged via TensorBoard’s
`add_hparams` call so each run records exactly which loss was used.

### What Configurable Loss Provided

By pulling the loss setup into a function the researchers paved the way for
systematic experiments without editing core training loops.  Every run now
captures its loss type alongside the learning rate and batch size, enabling clear
comparisons when reviewing results across dozens of job folders.

### Reproducing the Loss‑Function Update

After checking out this commit you can run either training script normally.
TensorBoard should show `lossFunc: RMSE` in the hyperparameter table.  Change the
`lossFunction` variable to `"MAE"` to verify that the helper swaps in the
appropriate criterion.

## Summary of the Fifty-Ninth Commit

This commit kicked off the first dedicated moment‑model experiment under the new
PyTorch pipeline.  `ss_torch_momentModel.sh` now writes its log to
`result_qsub/exp_1` so the output doesn’t mix with earlier trials.  Inside the
Python script the TensorBoard writers save runs under a nested `{dataType}`
folder, organizing angle and moment logs separately.

### What the First Moment Experiment Tried

The reconfigured log paths made it simpler to track metrics for this initial
PyTorch moment run.  While the network architecture stayed the same, moving logs
into a clean directory laid the groundwork for a series of controlled
experiments.

### Reproducing the First Moment Experiment

Submit the updated `ss_torch_momentModel.sh` to your queue.  When it finishes
check `result_qsub/exp_1` for the job output and look under
`logs/pytorch/.../moBWHT/train_*` to view the TensorBoard data.

## Summary of the Sixtieth Commit

For the second experiment the loss function was temporarily switched from RMSE
to mean absolute error.  The job script now targets `result_qsub/exp_2` and the
`lossFunction` variable in `torch_momentModel.py` is set to `"MAE"`.

### What the Second Moment Experiment Explored

Changing the loss allowed the team to gauge how sensitive the moment model was
to the choice of error metric.  Keeping the rest of the code identical ensured
that any differences in learning curves could be attributed to this single
change.

### Reproducing the Second Moment Experiment

Run the moment job script again after pulling this commit.  TensorBoard should
list `lossFunc: MAE` for each run under `exp_2`.

## Summary of the Sixty-First Commit

Experiment three adjusted several hyperparameters at once.  The moment model now
uses a learning rate of `0.0002` with a batch size of `32` and switches back to
RMSE as the loss.  Logs are saved under `result_qsub/exp_3` to preserve the
output separately from previous attempts.

### What the Third Moment Experiment Changed

By doubling the batch size and raising the learning rate slightly, the team
investigated whether training could converge faster without sacrificing
stability.  Recording these runs in a fresh directory kept the progression of
settings clear.

### Reproducing the Third Moment Experiment

Queue `ss_torch_momentModel.sh` one more time and inspect the new log folder.
The hyperparameter table should report `lr=0.0002`, `bsize=32`, and
`lossFunc: RMSE`.

## Summary of the Sixty-Second Commit

The fourth moment experiment extended the hyperparameter sweep by returning to mean absolute error while keeping the learning rate at `0.0002` and the batch size at `32`.  `ss_torch_momentModel.sh` now directs all output to `result_qsub/exp_4`, ensuring the results stand apart from the earlier runs.  In `torch_momentModel.py` the `lossFunction` variable was toggled back to `"MAE"`, leaving the rest of the training loop untouched.

### What the Fourth Moment Experiment Investigated

With the initial RMSE and MAE tests complete, this trial measured how MAE behaved under the more aggressive learning rate and batch size introduced in the previous commit.  By isolating the loss function as the only change, the team could directly compare how sensitive the network was to that choice while holding the optimizer settings constant.

### Reproducing the Fourth Moment Experiment

Submit `ss_torch_momentModel.sh` once more after pulling these changes.  When training finishes, inspect `result_qsub/exp_4` for the console output and review the TensorBoard logs under the matching `exp_4` directory.  The hyperparameter display should show `lossFunc: MAE` along with the unchanged learning rate and batch size.

## Summary of the Sixty-Third Commit

Attention then shifted to the angle model.  The job script was rewritten so angle experiments have their own numbered folders, starting with `result_qsub/exp_1`.  No parameters changed yet, so the Python script continues to train with a learning rate of `0.0001`, a batch size of `16`, and RMSE as the loss function.  This commit merely set up a clean baseline for subsequent angle trials.

### What the First Angle Experiment Established

Mirroring the first moment experiment, this run recorded the performance of the PyTorch angle model using default hyperparameters.  Separating the logs into their own directory keeps the angle metrics distinct from moment results and makes it easier to compare across later revisions.

### Reproducing the First Angle Experiment

Run `ss_torch_angleModel.sh` after applying the commit.  The job output appears under `result_qsub/exp_1` and the training progress can be inspected through the TensorBoard files generated in that folder.

## Summary of the Sixty-Fourth Commit

For the second angle experiment the loss function was switched to mean absolute error.  The job script increments the output folder to `result_qsub/exp_2` while `torch_angleModel.py` now sets `lossFunction = "MAE"`.  Learning rate and batch size remain at `0.0001` and `16` respectively.

### What the Second Angle Experiment Explored

This change isolated the effect of MAE on the angle predictions.  Comparing the new log against the baseline reveals whether absolute error provides a clearer signal than RMSE when training with the same optimizer settings.

### Reproducing the Second Angle Experiment

Submit the angle job script again so that a new set of logs appears in `result_qsub/exp_2`.  TensorBoard should list `lossFunc: MAE` for these runs.

## Summary of the Sixty-Fifth Commit

After reviewing the second trial the researchers re-ran the original angle experiment to double-check the results.  The job script reverts its output directory to `result_qsub/exp_1`, and `torch_angleModel.py` switches the loss function back to RMSE.  All other parameters stay the same.

### Why the First Angle Experiment Was Repeated

Because MAE produced unexpected behavior, the team wanted a direct comparison with a freshly generated RMSE run.  Recreating the baseline under identical conditions guards against misconfigured environments or transient cluster issues.

### Reproducing the Angle Experiment Re-Run

Launch `ss_torch_angleModel.sh` one last time.  New logs should overwrite the previous contents of `result_qsub/exp_1`.  Verify that the hyperparameters match the earlier baseline and that the metrics track closely to the first attempt.

## Summary of the Sixty-Sixth Commit

The third angle experiment adjusted the PyTorch training configuration to mirror the hyperparameter sweep performed on the moment model.  `torch_angleModel.py` now trains with a learning rate of `0.0002` and a batch size of `32`, while `ss_torch_angleModel.sh` writes its log files to `result_qsub/exp_3`.  No other code changed, but this allowed the researchers to see how doubling the batch size and increasing the learning rate influenced convergence when predicting angles.

### What the Third Angle Experiment Tested

By running with the same optimizer settings that had previously improved the moment model, the team hoped to shorten training time without sacrificing accuracy.  Keeping the architecture and dataset constant meant that any change in performance could be attributed to these two hyperparameters alone.  Recording the output in a new directory also provided a clean comparison against the earlier angle runs.

### Reproducing the Third Angle Experiment

After updating the repository to this commit, submit `ss_torch_angleModel.sh`.  The console output should appear under `result_qsub/exp_3`, and TensorBoard will log the new learning rate and batch size.  Compare these metrics to the baseline to assess whether the faster schedule helps or hurts.

## Summary of the Sixty-Seventh Commit

The fourth angle experiment kept the same higher learning rate and larger batch size but switched the loss function from root‑mean‑square error to mean absolute error.  `ss_torch_angleModel.sh` now points to `result_qsub/exp_4` and `torch_angleModel.py` sets `lossFunction = "MAE"`.

### What the Fourth Angle Experiment Examined

Changing only the loss metric allowed the researchers to isolate its effect while holding the optimizer settings steady.  They wanted to know if MAE would produce smoother gradients or lead to different convergence behavior compared to RMSE when the batch size and learning rate were already tuned upward.

### Reproducing the Fourth Angle Experiment

Run the angle job script again after pulling this commit.  New logs should populate `result_qsub/exp_4`.  TensorBoard will report `lossFunc: MAE` along with the unchanged learning rate and batch size.

## Summary of the Sixty-Eighth Commit

This commit reorganized the output directories and refined the timestamp format used when saving logs.  Separate `angle` and `moment` folders were created under `result_qsub`, and both job scripts were updated to write there.  The Python training scripts revert to the baseline hyperparameters—learning rate `0.0001`, batch size `16`, and RMSE loss—and now generate timestamps with microsecond precision using `datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-2]`.  A small notebook named `StudyRoom/datetime.ipynb` demonstrates the new formatting.

### Why Fine‑Grained Time Stamps and Folder Structure Matter

Organizing logs by model type keeps angle and moment experiments clearly separated, while the more detailed timestamps guarantee unique file names even when multiple jobs start within the same second.  These tweaks prevent accidental overwrites on the cluster and make it easier to trace results back to specific runs.

### Reproducing the New Timestamp Setup

With this commit checked out, launch either job script.  Observe that new directories such as `result_qsub/angle/exp_1` appear and that any generated TensorBoard folders include the microsecond timestamp.  The training parameters should match the original baseline settings.

## Summary of the Sixty-Ninth Commit

A major grid‑search framework was introduced to automate hyperparameter exploration.  Two new scripts, `grid_torch_angleModel.py` and `grid_torch_momentModel.py`, iterate over combinations of learning rate, batch size, and loss function.  Matching job files (`grid_ss_torch_angleModel.sh` and `grid_ss_torch_momentModel.sh`) submit six nine‑hour jobs in sequence.  A notebook called `grid_training.ipynb` documents the search logic.  The standard training scripts were also updated: they now print the selected parameters, log graphs under a revised directory layout, and store results in `result_qsub/angle/exp_5` by default.

### What the Grid Search Implementation Provided

By exploring multiple settings automatically, the team could identify promising hyperparameter values without manually editing the code for each run.  The grid scripts cycle through three learning rates, three batch sizes, and both RMSE and MAE losses, training a separate model for each combination across all folds.

### Reproducing the Grid Search Runs

Execute `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh` to launch the full sweep on the cluster.  Check `result_qsub/angle/grid` or `result_qsub/moment/grid` for the queued job outputs, and review the TensorBoard logs stored under `logs/pytorch/...` to compare each trial.


## Summary of the Seventieth Commit

The TensorBoard logs generated by the grid-search scripts were being overwritten whenever multiple folds started at nearly the same time.  Commit 70 corrected this by moving the timestamp generation inside the nested loops that iterate over hyperparameters and folds.  Both `grid_torch_angleModel.py` and `grid_torch_momentModel.py` now call `datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-2]` right before initializing the `SummaryWriter`.  The log directories were updated to follow the pattern `logs/pytorch/<modelVersion>/<nameDataset>/<dataType>/<numFold>_fold/train/<time>` so each fold stores its metrics under a unique subfolder.  This prevents clashes and ensures that every run can be inspected separately in TensorBoard.

### Why Timestamping Matters

During the initial grid search several folds would launch in quick succession.  Because the scripts previously set `time` only once, all writers for a given job attempted to use the same directory.  Moving the timestamp inside the loop guarantees that even two folds started within milliseconds of each other write to distinct locations.  The change also mirrors the directory hierarchy used by the single-run scripts, keeping the project structure consistent.

### Reproducing the Fixed Logging Behavior

After pulling this commit, execute `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh` again.  Inspect `logs/pytorch/...` and confirm that each fold has its own timestamped folder.  Opening TensorBoard should now show a separate run for every hyperparameter combination without files being overwritten.

## Summary of the Seventy-First Commit

Variable names inside the grid-search scripts were streamlined for clarity.  Dictionaries holding candidate values for learning rate, batch size, and loss function were renamed to `list_learningRate`, `list_batch_size`, and `list_lossFunction`.  The loops were adjusted accordingly so that each option is retrieved from these dictionaries before training.  The companion job scripts also changed their SGE job names to `Gtorch_angle` and `Gtorch_moment` to distinguish them from earlier experiments.

### Why the Renaming Was Important

Previous versions reused variable names in ways that occasionally caused confusion when parameters were reassigned during the loops.  Prefixing them with `list_` makes it clear that these are collections of possible values rather than the currently selected setting.  Updating the job names likewise helps when monitoring the cluster queue, as grid-search jobs can be easily filtered.

### Reproducing the Updated Scripts

Simply submit the grid job scripts as before.  The output should now reference the new variable names, and the queue will list jobs called `Gtorch_angle` or `Gtorch_moment`.  All other behavior remains the same, so existing instructions for supplying data and examining TensorBoard still apply.

## Summary of the Seventy-Second Commit

This commit made minor adjustments to the console messages printed during grid search.  The increment of the `count` variable, which tracks how many configurations have run so far, now occurs after the settings are displayed.  The redundant line printing just the count was removed.  As a result, each iteration outputs a single concise line describing the current data type, learning rate, batch size, loss function, model version, and dataset.

### Why Clean Output Helps

When dozens of jobs are launched automatically, clear logging is essential for diagnosing failures.  The previous two-line format sometimes led to mismatched counts if a run crashed early.  By updating the message and moving the counter increment, the logs more accurately reflect the order in which configurations were attempted.

### Reproducing the Tidier Console Messages

Run either grid job script and watch the terminal or qsub log.  You should see messages like `count:0 | 현재 설정 Type:angle, lr:0.0001, BS:16, LF:RMSE, modelV:V1, DataSet:IWALQQ_1st` with the count increasing after each combination completes.

## Summary of the Seventy-Third Commit

The TensorBoard directory structure was refined again so that hyperparameters are embedded directly in the folder names.  Instead of grouping runs solely by timestamp, the scripts now create writers under paths such as `logs/pytorch/<modelVersion>/<nameDataset>/<dataType>/LR_<learningRate>_BS_<batch_size>_LF_<lossFunction>/train/<numFold>_fold`.  This makes it trivial to compare metrics from different learning rates or loss functions within TensorBoard’s interface.

### Why Embedding Hyperparameters in Paths Helps

Grid searches generate a large number of runs.  Without descriptive folder names, it is easy to lose track of which settings produced which curves.  Encoding the learning rate, batch size, and loss function directly in the path lets collaborators open TensorBoard and immediately identify the conditions behind each experiment without cross-referencing log files.

### Reproducing the New Display Layout

After applying this commit, launch a grid search once more.  Navigate to the newly created directories under `logs/pytorch/` and verify that each one contains the hyperparameter values in its name.  When you start TensorBoard pointing at the parent directory, the sidebar will list runs labeled by their exact configuration, simplifying result analysis.


## Summary of the Seventy-Fourth Commit

The grid-search job scripts were updated to activate a new Conda environment named `torch`.  Earlier versions referenced `torchIMU`, but the researchers standardized their cluster setup and wanted all PyTorch jobs to rely on the same environment.  `grid_ss_torch_angleModel.sh` and `grid_ss_torch_momentModel.sh` now call `conda activate torch` after loading the `miniconda` module.  No Python code changed, but this ensures that the correct versions of PyTorch and its dependencies load consistently across all grid-search runs.

### Why the Environment Change Matters

The cluster originally contained several Conda environments with overlapping packages.  Misaligned paths occasionally caused runs to fail if the wrong environment was activated.  By settling on a single environment name and updating the job scripts accordingly, the team eliminated this source of confusion and made future updates easier to manage.

### Reproducing the Updated Job Scripts

Verify that your SCC account contains an environment called `torch` with the required packages (PyTorch, scikit-learn, etc.).  Then submit `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh`.  The logs under `result_qsub/*/grid` should show that the `torch` environment was activated before training began.

## Summary of the Seventy-Fifth Commit

All remaining job scripts received the same environment update.  The standard angle and moment jobs now activate a Conda environment named `scc` instead of the older `sccIMU`.  The lean versions and the single-run PyTorch scripts also switch from `torchIMU` to `torch`.  These edits affect `ss_angleModel.sh`, `ss_momentModel.sh`, `ss_lean_angleModel.sh`, `ss_lean_momentModel.sh`, `ss_torch_angleModel.sh`, and `ss_torch_momentModel.sh`.

### Why Consolidating the Environments Helps

Maintaining multiple nearly identical environments made it difficult to reproduce results.  By renaming them to `scc` and `torch` and updating every script, the authors created a consistent baseline that new contributors could follow.  It also reduced the risk of accidentally running with mismatched library versions when launching jobs from different folders.

### Reproducing the Revised Environment Setup

Create environments named `scc` and `torch` on your cluster account with the appropriate dependencies.  After applying this commit, submit any of the `ss_*` job scripts.  Confirm in the output that the new environment names appear just after the `module load miniconda` line.

## Summary of the Seventy-Sixth Commit

This commit overhauled the grid-search scripts to run an extended learning-rate sweep while fixing the loss function to mean absolute error.  The Python files `grid_torch_angleModel.py` and `grid_torch_momentModel.py` now define `list_learningRate` with six values ranging from `0.0001` to `0.008`.  Batch size is held constant at `32`, and only `MAE` is used for the loss.  The outer loops were rewritten to iterate over this new dictionary and the training loop was expanded with additional logging of per-axis nRMSE.  Companion notebooks in `StudyRoom/` (`datetime.ipynb` and `grid_training.ipynb`) document these changes and capture example output.

### What the Revised Grid Search Explores

By varying only the learning rate, the researchers could pinpoint its effect on convergence without the confounding influence of different batch sizes or loss functions.  The added nRMSE tracking provides more granular feedback during training, making it easier to judge which rate yields the best balance between speed and accuracy.

### Reproducing the Extended Sweep

Run `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh` after updating your environments.  Each script schedules six jobs, one for every learning rate in the list.  TensorBoard logs will appear under directories labeled `LR_<learningRate>_BS_32_LF_MAE`, letting you compare the resulting curves side by side.

## Summary of the Seventy-Seventh Commit

The final commit in this group fine-tuned the job scripts and Python files from the previous step.  Both grid-search job scripts now request ten hours of wall time instead of nine to accommodate the longer training loop.  Minor adjustments were made to the notebook outputs, and the Python scripts received small formatting tweaks while retaining the focus on learning-rate sweeps with MAE loss.  No new hyperparameters were introduced, but these tweaks ensured that all runs completed successfully on the cluster.

### Why the Extra Hour Was Needed

Early tests showed that some learning-rate combinations occasionally exceeded the nine-hour limit, causing the scheduler to terminate them prematurely.  Increasing the wall time to ten hours provided a safety margin so that every configuration could finish even on a busy node.

### Reproducing the Updated Timing

Submit either grid job script as before.  The `qsub` output should indicate a ten-hour reservation.  Once the runs complete, verify that the logs exist for all six learning rates without truncated epochs.

## Summary of the Seventy-Eighth Commit

This merge commit pulled in the latest changes from the remote `main` branch so
the grid-search experiments could continue on top of the most current code. No
files were modified directly, but synchronizing with the upstream repository was
important because earlier collaborators had tweaked some of the preprocessing
notebooks and environment configuration. Merging ensured that everyone was
working from a consistent baseline before the next round of hyperparameter
tests.

### Why Keeping the Branches in Sync Matters

Without this merge the local development branch would have diverged from the
repository hosted on GitHub. Any subsequent fixes from teammates could have led
to confusing conflicts. By incorporating the remote history at this stage the
researchers avoided surprises and made sure that the dataset paths and Conda
environment names matched those used by their colleagues.

### Reproducing the Merge State

Because this commit introduced no functional changes, simply pulling the
repository at this revision will place you on the same footing. All scripts and
notebooks will run exactly as they did after the previous commit, but the git
history now reflects the synchronized merge.

## Summary of the Seventy-Ninth Commit

While reviewing the grid-search results the authors discovered that the learning
rate dictionary listed `0.008` instead of the intended `0.0008`. This subtle
typo meant the search skipped over a promising setting. Both
`grid_torch_angleModel.py` and `grid_torch_momentModel.py` were corrected so the
entry now reads `0.0008`. Each dictionary still contains six options, ranging
from `0.0001` to `0.002`, but the fourth value is the newly fixed `0.0008`.

### Why a Single Digit Matters

Learning rate has a huge impact on convergence. The misplaced decimal point made
the model step ten times larger than expected, often causing divergence. By
correcting the value the researchers ensured the sweep covers a reasonable range
and that comparisons between rates are meaningful.

### Reproducing the Corrected Sweep

After checking out this commit, run either grid-search job script. The printed
parameter list should include `0.0008` among the learning rates. Review the
TensorBoard logs under `LR_0.0008_BS_32_LF_MAE` to verify that this setting now
produces sensible loss curves.

## Summary of the Eightieth Commit

To quickly verify the effect of the corrected learning rate, the team trimmed
the dictionaries so that only `0.0008` remains. This commit updates both grid
search Python files to define `list_learningRate = {0:0.0008}`. The job scripts
were left unchanged, meaning they still schedule six runs, but the Python code
now ignores those indices and repeatedly trains with the single specified rate.

### Why Focus on One Rate

Narrowing the sweep allowed the researchers to confirm that 0.0008 indeed worked
as expected without waiting for the other configurations. It served as a sanity
check before re-expanding the search space in later commits.

### Reproducing the Single-Rate Test

Execute `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh`. Even
though the scheduler will report six tasks, each will run with the same learning
rate of 0.0008. Inspect the output directories to confirm that the logs all share
this setting.

## Summary of the Eighty-First Commit

Final preparations for the 0.0008 experiment reduced the requested wall time in
both grid-search job scripts from ten hours to one hour and thirty minutes. This
shorter reservation better reflected the actual runtime of a single learning-rate
trial. The Python scripts still specify `list_learningRate = {0:0.0008}`, keeping
the focus on that solitary value.

### Why Shorten the Wall Time

Earlier tests showed that the simplified run finished well under two hours.
Reserving a full ten hours needlessly tied up cluster resources, so the scripts
were adjusted to a more appropriate limit. This made queueing easier during busy
periods while still providing enough time for the training loops to complete.

### Reproducing the Adjusted Jobs

Submit the grid-search scripts as before. The `qsub` output should now indicate a
1.5-hour reservation. When the jobs finish, verify that the resulting directories
contain logs for the 0.0008 learning rate and that each run completed without
timeout warnings.

## Summary of the Eighty-Second Commit

A brief merge synchronized the working branch with the upstream `main` repository once more.  No files were modified in this project directory, but the commit ensured that all collaborators shared the same history before launching additional experiments.

### Why Another Merge Was Needed

Even small differences between branches can cause troublesome conflicts later on.  By pulling the remote changes early the team kept the grid-search work compatible with improvements to the preprocessing notebooks and environment files being made by others.

### Reproducing the Merge State

Checkout this revision directly.  Because it contains no functional changes the scripts run exactly as they did after the Eighty-First Commit, but `git log` now shows the merge.

## Summary of the Eighty-Third Commit

With the learning-rate bug fixed the researchers broadened the sweep to explore combinations of three rates and three batch sizes.  Both Python grid scripts now define
`list_learningRate = {0:0.001, 1:0.002, 2:0.004}` and
`list_batch_size = {0:32, 1:64, 2:128}`.  Nested loops iterate over these dictionaries so each fold trains nine separate models.  The accompanying job scripts were updated to request an eighteen-hour wall time in anticipation of the heavier workload.

### Why Expand the Grid

Earlier tests suggested that the model responded differently to learning rate depending on batch size.  Sweeping over all nine combinations allowed the team to identify the most stable pairing without running individual jobs by hand.

### Reproducing the Expanded Search

Submit `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh` on your cluster.  The `qsub` output lists nine tasks and reserves eighteen hours.  When finished, check the TensorBoard logs under directories named with both the learning rate and batch size to compare results.

## Summary of the Eighty-Fourth Commit

During data curation an indexing error surfaced in `2_Data_PDFViewNCheck.py`.  The code that decided whether a file was marked `TBD` used an empty check that failed on NumPy arrays.  The condition was rewritten to compare the array values directly, preventing a crash when iterating through the exclusion list.  A small notebook named `test.ipynb` was added under `StudyRoom/` to demonstrate the fix.

### What the Bug Fix Solved

The script now properly pauses only on files labeled `TBD`, letting researchers quickly review questionable trials without skipping or repeating entries.  The test notebook walks through a short sample to confirm the logic works as expected.

### Reproducing the Check Script

Run `python 2_Data_PDFViewNCheck.py --mode CHK` after populating the dataset directory.  Step through the prompts shown in the notebook and verify that the program no longer throws an exception when it encounters `TBD` entries.

## Summary of the Eighty-Fifth Commit

Another merge from the remote repository pulled in fresh training notebooks and updated PyTorch scripts.  The added files include `StudyRoom/datetime.ipynb` and `StudyRoom/grid_training.ipynb`, both outlining how to schedule grid runs interactively.  Existing job scripts and training programs received minor tweaks during the merge to stay compatible with these notebooks.

### Why Incorporate the New Notebooks

The notebooks documented important setup details such as kernel selection and log organization.  Merging them ensured that anyone cloning the project could reproduce the grid-training workflow exactly as performed on the BU SCC.

### Reproducing the Merge Results

Check out this commit and open the new notebooks under `training/StudyRoom/`.  They should run without modification alongside the grid-search scripts introduced earlier.

## Summary of the Eighty-Sixth Commit

Following the merge, the team rebased the feature branch so that all grid-search improvements sat atop the latest history cleanly.  The rebase commit recreates the updated job scripts, Python training files, and notebooks in a single linear sequence.  It also tweaks the standalone `torch_angleModel.py` and `torch_momentModel.py` scripts to accept different loss functions and to timestamp TensorBoard directories with microsecond precision.

### What the Rebase Achieved

Consolidating the changes made it easier to share patches and ensured that subsequent commits applied cleanly.  The refined training scripts support both RMSE and MAE losses and store logs under directories named by experiment, dataset, and timestamp, simplifying later analysis.

### Reproducing the Rebased Setup

After cloning this revision, run any of the `ss_*` or `grid_*` job scripts.  They produce the same results as before the rebase, but the Git history is now linear and the log folders include high-resolution timestamps for easier sorting.

## Summary of the Eighty-Seventh Commit

After rebasing the grid-search work, the authors performed a simple merge to align the repository with GitHub once more.  This commit has no file changes of its own, but it records that the local history was reconciled with the upstream `main` branch.  Keeping the branch in sync ensured that later patches would apply cleanly without conflicts.

### Why the Extra Merge Was Necessary

The rebase from the previous commit rewrote local history.  Before pushing those updates back to GitHub, the developers merged the remote tip so that all collaborators shared the same ancestry.  Although the merge produced no code differences, it prevents divergent histories that could complicate future pulls.

### Reproducing the Merge State

No action is required beyond checking out this revision.  All scripts behave exactly as they did after the rebase, but `git log` now shows a merge commit connecting the histories.

## Summary of the Eighty-Eighth Commit

With the repository synchronized, the team launched a new learning‑rate experiment.  Both grid‑search Python scripts were updated so `list_learningRate` contains `{0:0.006, 1:0.008, 2:0.01}` while `list_batch_size` is simplified to `{0:128}`.  The accompanying job scripts reduce the wall time from eighteen hours to five hours since only three runs are queued.  MAE remains the loss function.

### What the New Hyperparameters Tried

Earlier tests indicated that extremely small learning rates might converge slowly.  This commit explores a more aggressive range from 0.006 to 0.01 with a large batch size of 128.  Shorter five‑hour reservations were sufficient because fewer combinations were involved.

### Reproducing the LR Test

Submit either `grid_ss_torch_angleModel.sh` or `grid_ss_torch_momentModel.sh`.  TensorBoard logs should appear under directories named with the chosen rate and a batch size of 128.  Examine the loss curves to judge whether these higher learning rates are stable.

## Summary of the Eighty-Ninth Commit

A subsequent merge brought in a small but important bug fix for the PDF review script.  The check for `TBD` files in `2_Data_PDFViewNCheck.py` now compares `list_Excluded_byFig` values directly using `.values == 'TBD'` instead of an empty test that failed on arrays.  A new notebook, `preperation/StudyRoom/test.ipynb`, demonstrates how the script iterates through the dataset and pauses on flagged entries.

### What the Notebook Demonstrated

The added notebook walks through a mini dataset so users can verify the revised logic.  It loads example file lists, runs the exclusion check, and shows that only items labeled `TBD` trigger the interactive prompt.  This confirms the fix works across pandas versions.

### Reproducing the Bug Fix

Run `python 2_Data_PDFViewNCheck.py --mode CHK` on a directory containing your raw files.  Compare the behavior with the cells in `test.ipynb` to ensure no exceptions occur when the exclusion list is empty.

## Summary of the Ninetieth Commit

While preparing additional datasets the notebook `4_DataSet_CAN_MYWAY.ipynb` failed to load because leftover merge markers corrupted its JSON structure.  This commit cleans the notebook by removing those markers, resets the random seed to 41, and switches the interpreter metadata to the `torch` environment.  The corrected file opens normally in Jupyter.

### Why the Notebook Was Broken

During an earlier merge, conflict markers such as `<<<<<<<` and `=======` were accidentally committed.  Jupyter could not parse the malformed file, preventing further edits.  Removing the markers and updating the kernel information restored full functionality.

### Verifying the Fix

Open `preperation/4_DataSet_CAN_MYWAY.ipynb` after checking out this commit.  The notebook should load without errors and display the proper Python 3.8.13 (`torch`) kernel specification.

## Summary of the Ninety-First Commit

To support forthcoming autoencoder experiments the developers created `4_DataSet_IWALQQ_AE.ipynb`.  This extensive notebook assembles IMU signals into a four‑dimensional array shaped `(N, row, column, channel)` suitable for convolutional models.  It documents the calibration steps applied to the `IWALQQ_1st_correction` data and saves the resulting tensors for reuse.

### Why This Dataset Was Needed

Autoencoder models require raw sensor sequences rather than aggregated features.  By reorganizing the data into a consistent [batch, time, feature, channel] format, this notebook lays the groundwork for experimenting with AE and VAE architectures while preserving the calibrated orientation from earlier preprocessing.

### Reproducing the Dataset Creation

Launch the notebook and execute each cell in order.  Provide paths to the corrected IMU files and specify an output directory.  When finished you will obtain an `IWALQQ_AE_1st` dataset that mirrors the one referenced in later training scripts.


## Summary of the Ninety-Second Commit

To better understand preprocessing effects, a new notebook `preperation/StudyRoom/diffScalingMethod.ipynb` compares feature-wise scaling with sensor-wise scaling on the autoencoder dataset. Each strategy rescales the IMU channels differently before they feed into the model.

### What the Scaling Notebook Explores

The notebook loads `IWALQQ_AE_1st` and applies both normalization schemes. Feature-wise scaling treats every feature across all sensors the same, while sensor-wise scaling normalizes each axis individually. Histograms and summary statistics reveal how these choices change the value ranges seen by the networks.

### Reproducing the Scaling Comparison

Open `diffScalingMethod.ipynb` and run all cells. Provide paths to the AE dataset when prompted. The final plots let you judge which scaling approach preserves signal structure best.

## Summary of the Ninety-Third Commit

Documentation inside the dataset notebooks was refined. `4_DataSet_CAN_MYWAY.ipynb` and `4_DataSet_IWALQQ_AE.ipynb` now contain clearer comments and reorganized section headers. No computations changed, but the narrative explains every preprocessing step in greater detail.

### Why the Notebook Comments Matter

Because these notebooks guide others through dataset creation, precise annotations are essential. The updated explanations clarify variable names and the order in which files are loaded and transformed.

### Updating the Notebooks

Simply open the revised notebooks to read the improved commentary. Running them yields the same output datasets as before.

## Summary of the Ninety-Fourth Commit

With the notebooks polished, the authors generated the official `IWALQQ_AE_1st` dataset. The creation steps recorded in `4_DataSet_IWALQQ_AE.ipynb` gather the corrected IMU signals into a four-dimensional tensor and write it to disk. Outdated exploratory code was removed from `diffScalingMethod.ipynb` once this dataset was finalized.

### What the Notebook Produces

Executing the notebook loads each subject's calibrated sequences, stacks them by trial and sensor channel, and saves a compressed `.npz` archive. The result serves as the starting point for subsequent autoencoder experiments.

### Reproducing the AE Dataset

Run every cell in `4_DataSet_IWALQQ_AE.ipynb`, adjusting the input and output directories for your environment. The resulting file should match the `IWALQQ_AE_1st` dataset referenced by later training scripts.

## Summary of the Ninety-Fifth Commit

Next, the grid-search scripts were modified to train dense networks on the new autoencoder data. Their dataset classes now reshape the `(N, time, feature)` tensors into flat vectors before converting them to PyTorch tensors. Additional cells in `diffScalingMethod.ipynb` illustrate this reshaping step.

### How the Scripts Adapt AE Data

By calling `np.reshape` inside the dataset loader, each sample becomes a single long vector that existing dense architectures can consume without modification. This allows quick experiments with the AE data while reusing earlier model definitions.

### Reproducing the Dense AE Training

Generate `IWALQQ_AE_1st` first, then execute `grid_torch_angleModel.py` or `grid_torch_momentModel.py`. They will load the AE dataset, reshape it on the fly, and proceed through the same grid-search routine used for previous datasets.

## Summary of the Ninety-Sixth Commit

Finally, the single-setting shell scripts were updated to test the autoencoder data. Their comments now note that only three jobs are submitted instead of nine, reflecting the reduced hyperparameter grid.

### What the Shell Scripts Run

`grid_ss_torch_angleModel.sh` and `grid_ss_torch_momentModel.sh` still request five-hour SCC slots, but each now launches just three learning-rate combinations. This quick check verified that `torch_Dense_1st` trains correctly with the AE-formatted inputs.

### Running the AE Test

After preparing the dataset and scripts, submit either shell file with `qsub`. TensorBoard logs will appear under `result_qsub/angle/grid` or `result_qsub/moment/grid`, confirming that the dense model can ingest the autoencoder-style data.

## Summary of the Ninety-Seventh Commit

The ninety-seventh commit corrected a subtle oversight in the grid-search scripts. The previous commit still referenced the earlier `IWALQQ_1st_correction` dataset even though the autoencoder work introduced a new tensor format named `IWALQQ_AE_1st`. Both `grid_torch_angleModel.py` and `grid_torch_momentModel.py` were updated so `nameDataset` points to this new dataset and `exp_name` reflects the May 11th experiments. Without this fix, the grid search would have continued training on the wrong data and produced incomparable results.

### Why the Dataset Name Matters

Switching the dataset variable ensures that every run now loads the four-dimensional AE tensors instead of the older three-axis arrays. This change is crucial for reproducing the experiments described in the preceding commits, where the autoencoder representation was introduced.

### Reproducing the Corrected Grid Search

Checkout this commit and open either grid-search script. Verify that `nameDataset` is set to `IWALQQ_AE_1st` and that `exp_name` equals `torch_20220511`. Run `python grid_torch_angleModel.py` or `python grid_torch_momentModel.py` to launch the revised experiments. They will create result folders tagged with the new experiment name and read data from the AE dataset.

## Summary of the Ninety-Eighth Commit

After switching datasets, the developers discovered that the normalized RMSE metric no longer worked. The AE tensors are scaled sensor‑wise across all axes, but the earlier metric routine assumed axis‑specific scaling parameters. To address this mismatch the authors rewrote the helper function `nRMSE_Axis_TLPerbatch` so it rescales predictions and targets using a single `MinMaxScaler` instance for the entire sensor vector. A small `MinMaxScalerSensor` class reshapes tensors before calling the standard scikit‑learn methods, allowing the scaler to fit and transform multi-dimensional arrays.

### How the Metric Works Now

Inside each training loop, the script loads the stored scaler and converts its arrays into PyTorch tensors residing on the active device. The metric function then removes the scaling from each batch, computes the per-axis RMSE in original units, and averages the scores across the batch. By eliminating axis indexing and moving the tensor conversion to the GPU, the metric now supports both feature‑wise and sensor‑wise scaled datasets without error.

### Replicating the Metric Update

Run either grid-search script from this commit onward. Training will print three nRMSE columns—one for each anatomical axis—without raising device or shape errors. Compare the results to earlier logs to confirm that the scaling change yields consistent values.

## Summary of the Ninety-Ninth Commit

During early runs with the revised metric, PyTorch reported a device mismatch when transferring scaler values to the GPU. The ninety-ninth commit patches this issue by explicitly creating the tensors on the same device as the model. The shell scripts `grid_ss_torch_angleModel.sh` and `grid_ss_torch_momentModel.sh` also gained an `LD_LIBRARY_PATH` export so CUDA libraries resolve correctly inside the new conda environment.

### Why This Fix Was Necessary

Without the explicit device placement, the training loop attempted to perform operations between CPU and GPU tensors, triggering runtime errors. Updating the shell scripts guarantees that the necessary shared libraries are visible when the job scheduler initializes the environment on the SCC cluster.

### Verifying the Device Fix

Submit either single-setting shell script with `qsub` after activating the `torch` environment. The job should now start successfully and write logs to `result_qsub` without crashing due to device mismatches.

## Summary of the One Hundredth Commit

To further debug the environment issues, the team created a small diagnostic script named `envChecker.py` under `training/StudyRoom`. This script imports PyTorch and related libraries to confirm that CUDA can be located from within the freshly created `imu` conda environment. The commit also mirrors the previous device-placement fix within `grid_torch_angleModel.py` so both angle and moment models apply the same tensor conversions.

### Using the New Environment

After cloning the repository, create the `imu` environment according to the project’s conda requirements. Run `python envChecker.py` to verify that PyTorch loads and detects the GPU. If the check passes, execute `grid_torch_angleModel.py` to perform a short training run using the AE dataset. Logs should appear under `result_qsub/angle/grid` just as they did for the moment model.

