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


## Summary of the One Hundred First Commit

The one hundred first commit tested the freshly created `imu` conda environment. Both `grid_ss_torch_angleModel.sh` and `grid_ss_torch_momentModel.sh` were modified so that the job scripts activate this new environment instead of `torch`. The angle-model script also shortened the wall time to 1.5 hours, renamed the job to `Genv_imu`, and redirected output logs to a temporary folder named `result_qsub/angle/grid_test_env_imu`. Correspondingly, `grid_torch_angleModel.py` updated the `exp_name` variable to `torch_test_env_imu` so that all logs and checkpoints reflect the test run.

### Why the Environment Needed Testing

After introducing `envChecker.py` in the previous commit, the developers wanted to confirm that a minimal environment containing only the necessary PyTorch packages could execute the grid-search jobs. This commit serves as a debugging snapshot: the scripts were intentionally tweaked to run a short training session while capturing every library path and CUDA interaction.

### Reproducing the Environment Test

1. Create the `imu` conda environment as described earlier in the README.
2. From this commit, submit `grid_ss_torch_angleModel.sh` via `qsub`. Verify that the job name appears as `Genv_imu` and that results write to `result_qsub/angle/grid_test_env_imu`.
3. Inspect the generated logs to ensure the model trains without missing library errors. The experiment should complete quickly thanks to the reduced wall time.

## Summary of the One Hundred Second Commit

With the environment confirmed, the second commit in this trio revisited the dataset notebooks. `preperation/4_DataSet_IWALQQ_AE.ipynb` and `preperation/StudyRoom/diffScalingMethod.ipynb` were both re-executed so that every cell ran from a clean state. Execution counts were reset and new output cells show that the AE dataset is generated correctly. The diff primarily reflects updated widget IDs and cell numbers, but the key result is a verified set of `.npz` files in `preperation/SAVE_dataSet/IWALQQ_AE_1st`.

### Confirming Dataset Generation

Open `4_DataSet_IWALQQ_AE.ipynb` and run all cells. You should see printed lists of source CSV files followed by confirmation messages once the dataset arrays are saved. Likewise, the `diffScalingMethod.ipynb` notebook now demonstrates how various scalers affect the data distribution. These notebooks document the preprocessing pipeline used for the AE experiments.

## Summary of the One Hundred Third Commit

After reviewing the short test run, the developers reverted the job-script tweaks. This commit completely undoes the changes from the one hundred first commit. The angle-model grid script again requests a five-hour wall time, activates the original `torch` environment, and writes output under `result_qsub/angle/grid`. The `exp_name` variable in `grid_torch_angleModel.py` returns to `torch_20220511`, and the LD_LIBRARY_PATH line in the moment-model script is restored.

### Why the Revert Happened

The quick `imu` environment test proved that CUDA worked correctly, so the team opted to continue using the stable `torch` environment for subsequent runs. Reverting ensures that longer experiments use the established configuration without the temporary debug paths.

### Getting Back to the Standard Setup

Simply pull this commit and resubmit your grid-search jobs. They will behave exactly as before the environment test, writing logs to the usual folders while training on the AE dataset for up to five hours per run.

## Summary of the One Hundred Fourth Commit

The one hundred fourth commit finalized the scaling comparison notebook. The team reran
`preperation/StudyRoom/diffScalingMethod.ipynb` so that every plot rendered correctly
under the new `imu` conda environment. The updated metadata now points to this kernel
and the execution counts reset to reflect a clean run. Minor cell edits tweak the
drawing commands, producing the final graphs used to assess sensor-wise versus
feature-wise normalization.

### Why the Notebook Was Rerun

Earlier attempts at the scaling comparison were executed under an older conda
environment.  When the developers consolidated around the `imu` environment they
discovered that some matplotlib settings and cached outputs no longer matched. By
rerunning the notebook from start to finish they ensured that all figures were
generated with identical library versions and that every plot accurately reflected
the preprocessing code included in the repository.

### Finishing the Scaling Comparison

Open `diffScalingMethod.ipynb` with Jupyter and execute the entire notebook. You should see side-by-side plots showing how each scaling approach affects the distribution of IMU signals. The notebook uses the `imu` kernel, so ensure that environment is active before launching Jupyter. Saving the notebook will embed the refreshed plots just as they appear in the repository.

## Summary of the One Hundred Fifth Commit

With the exploratory figures complete, the fifth commit reorganized the project
structure. Training scripts were moved into a new hierarchy under
`training/MODEL`. Separate subfolders now hold the original Keras dense models
(`kerasDense`), the PyTorch implementations (`torchDense`), and the grid-search
utilities. Because these scripts sit one level deeper, all job files were
updated to write logs using `../../result_qsub` paths. The commit also added
`preperation/StudyRoom/diffScaling_senwiseVSfeaturewise.png`, an image generated
from the prior notebook that illustrates the scaling comparison.

### Why the Reorganization Was Needed

As the repository grew it became difficult to track which scripts corresponded to
each framework and experiment type. Consolidating everything under the
`training/MODEL` directory groups related files together and prevents the root
folder from becoming cluttered. It also standardizes relative imports so that
notebooks and qsub scripts can locate the Python modules regardless of where the
repository is cloned.

### Adapting to the New Layout

After pulling this commit, use the paths under `training/MODEL` when submitting jobs. For instance, run `qsub training/MODEL/grid_torchDense/grid_ss_torch_angleModel.sh` to launch the PyTorch grid search. Logs will appear in the same `result_qsub` folders as before thanks to the updated relative paths. Scripts that import `angle_Model.py` or `moment_Model.py` must reference the new `training/MODEL/kerasDense` directory.

## Summary of the One Hundred Sixth Commit

The next commit refined these location changes and standardized how paths are defined inside the Python trainers. Hard-coded relative references were replaced with absolute directories pointing to the SCC project space. New variables `absDataDir`, `SaveDir`, and `logDir` now store these base paths so every script writes models and TensorBoard logs in a consistent location. The job scripts switched to activating the `imu` environment instead of `torch`, and the PyTorch code prints the chosen hyperparameters at startup.

In addition, `moment_Model.py` received significant cleanup. The metric functions now rely on the angle scalers for rescaling, ensuring that all three axes are evaluated uniformly. Learning rate and batch size defaults were adjusted to `0.0005` and `64`, and the loss function changed from MAE to RMSE. Matching updates in `torch_momentModel.py` keep the two implementations in sync.

### Why Standardizing Paths Matters

Early versions of the code assumed the repository was cloned in a specific
directory relative to the job scripts.  This led to broken imports whenever the
project was moved or shared with collaborators.  By defining absolute base paths
at the top of each trainer, the team ensured that checkpoints and logs always
save to the intended SCC locations regardless of the current working directory.
Switching the job scripts to the `imu` environment further consolidates all runs
under a single conda setup.

### Running with Absolute Paths

Check that the variables at the top of each training script point to your own project directories or leave them as the provided SCC paths. Activate the `imu` environment and submit the grid-search or single-run job scripts from their new locations. TensorBoard logs will accumulate under `/restricted/project/movelab/bcha/IMUforKnee/training/logs` while trained models save to `/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel`. These changes make it easier to resume experiments after moving the repository.

## Summary of the One Hundred Seventh Commit

The one hundred seventh commit completed the file reorganization effort by moving every training script—including both Keras and PyTorch versions—under the `training/MODEL` hierarchy. The grid-search shell scripts now reside in `MODEL/grid_torchDense`, while the single-run jobs for the dense networks live in `MODEL/kerasDense` and `MODEL/torchDense`. Paths inside these scripts were rewritten so logs are emitted to `../../result_qsub`. All Python trainers now define `absDataDir`, `SaveDir`, and `logDir` variables, ensuring dataset loading, checkpoint storage, and TensorBoard output work from any working directory. The job scripts likewise activate the `imu` environment before launching training.

### Why the Final Move Was Necessary

Prior commits only partially relocated the scripts, leaving some files in their old locations. As experiments grew this patchwork caused confusion when submitting jobs on the BU SCC cluster. Consolidating everything under one folder keeps imports consistent and avoids path errors. The absolute-base-directory approach also allows collaborators to clone the repository anywhere without editing numerous relative paths.

### Reproducing the Updated Runs

After syncing this commit, submit jobs from the new directories—for example `qsub training/MODEL/grid_torchDense/grid_ss_torch_angleModel.sh`. Verify that the path variables point to valid locations on your system. Activate the `imu` conda environment before launching training so that all dependencies load correctly. Result logs will appear in the top-level `result_qsub` folders as before, just referenced with an extra `../` to reach that directory.

## Summary of the One Hundred Eighth Commit

This revision merely merged in changes from the upstream repository. No files were modified beyond the automatic merge, so reproducing the project at this point simply requires pulling the commit.

## Summary of the One Hundred Ninth Commit

In the one hundred ninth commit the developers experimented with the Sequitur library to build an LSTM autoencoder. A new notebook named `training/MODEL/Sequitur/LSTM_AE_1stTry.ipynb` loads the AE dataset, constructs a small `LSTM_AE` model, and runs a few test cells. The notebook records commands to import Sequitur and prepare tensors, demonstrates passing a single sequence through the encoder and decoder, and prints the resulting shapes. Although the idea was promising, the team noted that the approach ultimately failed and the experiment was not continued.

### Why Sequitur Was Explored

Sequitur offers ready-made implementations of sequence autoencoders which could accelerate IMU reconstruction research. The notebook documents the preliminary steps for applying Sequitur to the existing dataset. However, compatibility issues prevented a successful training run, so this commit serves mainly as a record of the attempt.

### Trying the Notebook Yourself

Install Sequitur inside the `imu` environment using `pip install sequitur` and open the notebook in Jupyter. Execute the cells sequentially, modifying the absolute paths if your dataset resides elsewhere. The code will load one fold of the AE data, instantiate an `LSTM_AE` with a 42-dimensional input and a seven-unit latent vector, and attempt to process the tensors. Because the original authors did not finish debugging the workflow you may encounter errors, but the notebook provides a starting point for further Sequitur experiments.

## Summary of the One Hundred Tenth Commit

The repository next explored variational autoencoders built with TensorFlow.  A new folder `training/MODEL/Tensorflow_VAE_LSTM` contains a sample traffic‑volume dataset named `Metro_Interstate_Traffic_Volume.csv.gz` and a companion notebook `VAE_TimeSeries.ipynb`.  The notebook walks through loading this hourly traffic log, filling missing timestamps, and normalizing the features before constructing an LSTM‑based VAE.

### Why Experiment with a TensorFlow VAE?

Autoencoders were a natural next step after the supervised models because they can learn compact representations without labeled outputs.  TensorFlow was chosen for this first attempt since the team already used Keras for earlier scripts, making it straightforward to prototype a VAE with minimal boilerplate.  Training on a known time‑series dataset let them validate the architecture before turning to the IMU signals.

### Why Use a Traffic Dataset?

Before applying autoencoders to IMU sequences the team practiced on a well‑known time‑series problem.  The Minneapolis highway traffic data offers thousands of continuous measurements that are easy to visualize and require minimal preprocessing.  By training the VAE to reconstruct traffic counts they confirmed the architecture and Keras implementation worked as expected.

### Reproducing the VAE Demonstration

Decompress the CSV file if your environment does not handle gzip automatically, then open `VAE_TimeSeries.ipynb` under the new folder.  Execute each cell to load the data, define the encoder and decoder networks, and train the model.  The notebook saves plots of the reconstruction error so you can verify that the VAE learns meaningful patterns before moving on to the IMU dataset.

## Summary of the One Hundred Eleventh Commit

Having proven the basic concept, the authors refactored their PyTorch utilities into a small package called `CBDtorch`.  This commit introduces numerous modules under `training/MODEL/Pytorch_AE_LSTM`, including dataset loaders, metric functions, a `MinMaxScalerSensor` helper, and an `RMSELoss` class.  A simple `dirs.py` module provides path helpers, and `setup.py` builds the package so it can be installed locally.  Prebuilt `.egg` archives show that the team experimented with versioning the library.


A notebook at `training/MODEL/Pytorch_AE_LSTM/StudyRoom/AE_LSTM.ipynb` demonstrates how to import these modules to train an LSTM autoencoder on the IMU dataset.  Packaging the code in this way paved the road for cleaner experiments and easier reuse across notebooks.

### Why Build the `CBDtorch` Package?

As the number of experimental notebooks grew, maintaining duplicate helper functions became unwieldy.  Turning the most useful routines into an installable package ensured that every script imported the exact same dataset loaders and metric definitions.  This structure also mirrored how mature research codebases share common utilities, preparing the project for larger collaborations.

### Installing the Library

Navigate to `training/MODEL/Pytorch_AE_LSTM` and run `pip install -e .` to place `CBDtorch` on your Python path.  Afterwards you can open the accompanying notebook and follow the examples without needing to adjust import statements.  The included egg files are optional but illustrate how the package was intended for distribution.

## Summary of the One Hundred Twelfth Commit

Development of the autoencoder continued with a bidirectional LSTM architecture.  The `AE_LSTM.ipynb` notebook gained several new cells that define a BiLSTM encoder and decoder, leveraging the modules from `CBDtorch` for dataset handling and loss computation.  Training loops now summarize the model with `pytorch_model_summary` and log progress to TensorBoard.

### Why Add a BiLSTM?

Early experiments used a single-directional LSTM which only looked forward in time.  The researchers hypothesized that considering both past and future context would yield more accurate reconstructions of cyclical IMU signals.  A BiLSTM processes the sequence from both directions simultaneously, capturing patterns that span across the gait cycle.  This commit implemented that idea so its impact could be tested in practice.

### Running the BiLSTM Autoencoder

With `CBDtorch` installed, open the updated notebook and execute the cells in order.  The script will load the prepared IMU sequences, construct the BiLSTM autoencoder, and begin training.  Watch the reconstruction loss and TensorBoard graphs to gauge convergence.  These additions round out the initial exploration of sequence autoencoders before the project shifted back to supervised models.


## Summary of the One Hundred Thirteenth Commit

The next revision revisited the autoencoder notebook to demonstrate a complete training function.  Inside `training/MODEL/Pytorch_AE_LSTM/StudyRoom/AE_LSTM.ipynb` the authors refactored the example code so epochs are executed by a reusable `train_loop` routine.  This helper accepts a model, data loader, optimizer, and loss object, then iterates over the dataset while logging progress with TensorBoard.  Clearing the old execution counts keeps the notebook tidy, and a few cells were deleted to streamline the workflow.  A companion notebook named `torch_load_modelNvisualize.ipynb` gained minor updates illustrating how to reload a saved autoencoder and plot reconstruction results.

### Why Add a Training Loop Example?

Earlier experiments relied on ad-hoc loops scattered across notebooks.  By packaging the update/forward steps into a single function, researchers can quickly test different architectures without rewriting boilerplate each time.  The self-contained routine also mirrors patterns seen in larger PyTorch projects, preparing the codebase for more complex experimentation.

### Reproducing the Training Function

Ensure `CBDtorch` is installed as described in the previous section.  Open `AE_LSTM.ipynb` and run all cells.  The notebook will build the encoder and decoder, instantiate the optimizer, and call `train_loop` for the desired number of epochs.  After training completes, switch to `torch_load_modelNvisualize.ipynb` to verify that saved weights load correctly and that the reconstruction plots resemble those committed in the repository.

## Summary of the One Hundred Fourteenth Commit

This small change extended `.gitignore` to exclude the `training/MODEL/Pytorch_AE_LSTM/StudyRoom/data` directory.  The folder stored temporary datasets generated while iterating on the notebooks and could easily grow to hundreds of megabytes.  By ignoring it, the developers prevented accidental commits of large binaries and kept the repository lean.

### Why Ignore the StudyRoom Data Folder?

The experiments required frequent tweaks to preprocessing steps.  Rather than recomputing the IMU datasets every time, the authors cached intermediate files under `StudyRoom/data`.  These artifacts were specific to each machine and not needed for version control, so the ignore rule ensures collaborators do not accidentally upload them.

### Checking the Ignore Rule

After pulling this commit, create a dummy file inside `training/MODEL/Pytorch_AE_LSTM/StudyRoom/data` and run `git status`.  Git should report no changes, confirming that the pattern works.  You can safely store temporary tensors or CSV files here without affecting the repository history.

## Summary of the One Hundred Fifteenth Commit

Work on unsupervised models continued with a new notebook titled `AE_VAE.ipynb`.  This document experiments with both autoencoders and variational autoencoders in a single workflow.  It provides template cells for constructing the encoder, decoder, and latent sampling logic, along with placeholders for loading IMU sequences.  The commit message notes that development would proceed on a local machine, suggesting the notebook served as a sandbox before queuing longer runs on the BU SCC cluster.

### What the AE_VAE Notebook Demonstrates

By juxtaposing AE and VAE implementations side by side, the notebook clarifies the additional components required for a variational model—namely the reparameterization trick and KL‑divergence term.  Users can toggle between architectures to compare reconstruction quality and latent-space distributions.

### Running the Combined AE/VAE Notebook

Open `AE_VAE.ipynb` under the `StudyRoom` folder and follow the marked sections to load your data, define the networks, and launch training.  Because this commit predates finalized scripts, you may need to adjust file paths or hyperparameters manually.  The notebook is intended as a starting point for local experimentation before scaling up to cluster jobs.


## Summary of the One Hundred Sixteenth Commit

The next revision expanded the `AE_VAE.ipynb` notebook with additional cells that highlight the implementation differences between a standard autoencoder (AE) and its variational counterpart (VAE).  New markdown explanations walk through the encoder, decoder, and sampling steps side by side so readers can see where the VAE introduces stochasticity via the reparameterization trick.  The examples continue to rely on locally cached IMU tensors, reinforcing that these notebooks were still exploratory rather than part of the automated training pipeline.

### Why Document AE vs. VAE Differences?

Earlier commits introduced both model types but did not clearly explain how their architectures diverge.  By placing annotated code snippets in a single notebook, the authors created a teaching tool for new collaborators experimenting with unsupervised learning on time‑series data.  Comparing the loss functions and latent sampling strategies helps demystify variational methods before they are applied at scale.

### Trying the Notebook Yourself

Navigate to `training/MODEL/Pytorch_AE_LSTM/StudyRoom` and open `AE_VAE.ipynb`.  Run through the cells to review the side‑by‑side implementations.  You can manually tweak layer sizes or optimizer settings to observe how reconstruction loss behaves for each approach.  Because the dataset location is not hard‑coded, be prepared to point the notebook at your own IMU files if you wish to reproduce the exact experiments.

## Summary of the One Hundred Seventeenth Commit

Building on the previous exploration, this commit reorganized the StudyRoom materials.  The original notebook was renamed to `AE&VAE.ipynb` and a new `VAE_LSTM.ipynb` was added.  The latter contains a complete variational LSTM model ready for training, although the commit message notes that execution was not fully tested.  Two animated GIFs—`ae.gif` and `vae.gif`—were also included to visualize how each network processes sequences over time.

### What Was Completed

The authors finalized code cells for constructing both AE‑LSTM and VAE‑LSTM architectures, leveraging the utility modules introduced earlier.  Data‑loading placeholders remind users to supply their own IMU tensors, and training loops reference the reusable `train_loop` pattern.  Although these notebooks had not yet been run end to end, they provided a blueprint for subsequent cluster jobs.

### Why Reorganize the StudyRoom and Add Animated Visuals?

Splitting the material into two notebooks made it easier for collaborators to focus on either a straightforward autoencoder or a full variational model. Renaming the original notebook eliminated confusion about its contents, while `VAE_LSTM.ipynb` created a dedicated space for sequence-generation tests. The GIFs provide an at-a-glance demonstration of how the models process IMU sequences, helping new users grasp the purpose of each training cell before running the code.

### Reproducing the Draft Models

Open `AE&VAE.ipynb` first to review the combined AE and VAE examples.  Then work through `VAE_LSTM.ipynb`, supplying a small dataset to confirm that the model compiles and begins training.  The GIFs can be displayed within the notebooks to get an intuitive feel for sequence reconstruction versus generation.  If you encounter missing paths, adjust them to match your local environment before executing the cells.

## Summary of the One Hundred Eighteenth Commit

The final commit in this sequence refined the `setup.py` script within the `CBDtorch` package.  A brief comment now explains how to install the library using `python setup.py install`, clarifying the required command for users unfamiliar with editable mode.  Minor formatting tweaks round out the change.

### Why Clarify the Setup Instructions?

During earlier commits the package could only be imported if users manually added its directory to `PYTHONPATH`. Highlighting the standard `python setup.py install` command removes that confusion and ensures reproducible environments on the SCC cluster as well as local machines.

### Installing the Custom Library

From within `training/MODEL/Pytorch_AE_LSTM`, run `python setup.py install` to place `CBDtorch` into your Python environment.  Once installed, all notebooks and scripts that import modules from this package will function without additional path adjustments.  This step ensures that the autoencoder notebooks added in the preceding commits can locate the shared dataset loaders and metric functions.
## Summary of the One Hundred Nineteenth Commit

The project took its first steps toward running a sequence-based variational autoencoder on the cluster.  Two notebooks, `AE_LSTM.ipynb` and `VAE_LSTM.ipynb`, were expanded with cells showing how to launch experiments via qsub.  A new training script, `torch_angle_VAE_LSTM.py`, set up a grid-search loop over learning rates, batch size, and embedding dimension.  Supporting these experiments is a standalone module `vaelstm.py` that implements the encoder, decoder, and overall VAE model in PyTorch.

### Why Enable a VAE-LSTM Qsub Workflow?

Earlier autoencoder trials processed entire sequences with dense layers.  By introducing an LSTM-based variational autoencoder, the team could capture temporal dependencies in the IMU signals and produce a compact latent vector.  Integrating a qsub-compatible Python script meant these heavier models could be trained on BU's SCC without manual notebook execution.

### Reproducing the VAE-LSTM Experiments

Navigate to `training/MODEL/Pytorch_AE_LSTM` and submit `torch_angle_VAE_LSTM.py` using your own qsub wrapper or by running it interactively.  Adjust the `absDataDir`, `SaveDir`, and `logDir` variables inside the script so they point to your dataset and output locations.  The script iterates over predefined hyperparameters and logs losses to TensorBoard under `logDir`.  Review the generated logs to confirm that each fold trains successfully.

## Summary of the One Hundred Twentieth Commit

To simplify dataset handling, this commit refactored the loader code into a dedicated `dataset.py` file inside the `CBDtorch/custom` package.  The functions for assembling training and test tensors were pulled out of individual notebooks and scripts so they could be imported wherever needed.  The new helper returns PyTorch `Dataset` objects that supply batches for either prediction models or autoencoders.

### Why Split Out Dataset Functions?

As more models shared common preprocessing steps, duplicating the loading logic became error prone.  Consolidating these routines guarantees consistent scaling and fold selection across every experiment.  It also makes the VAE-LSTM script much cleaner because all file I/O is contained in one module.

### Using the Shared Dataset Loader

After installing `CBDtorch`, simply call `Dataset4predictor` or `Dataset4autoencoder` from `CBDtorch.custom.dataset`.  Provide the dataset directory, data type (`angle` or `moBWHT`), the session (`train` or `test`), and the fold number.  Each class returns tensors already cast to `float32` so they can be fed directly into PyTorch dataloaders.

## Summary of the One Hundred Twenty-First Commit

The final commit in this trio polished the VAE-LSTM pipeline.  The main training script was renamed from `torch_angle_VAE_LSTM.py` to `torch_VAE_LSTM.py` and trimmed of redundant comments.  Minor adjustments to the optimizer and save paths completed the refactor, establishing a stable entry point for future sequence models.

### Why Finalize the VAE-LSTM Script?

After verifying that the model trained correctly on a small subset, the developers standardized the filename and cleaned up the code so collaborators could reproduce the results.  The renamed script reflects that it can handle both angle and moment data, depending on the input arguments.

### Running the Updated Script

Submit `torch_VAE_LSTM.py` through your cluster's job scheduler or run it locally with CUDA enabled.  The script loads datasets via the refactored `CBDtorch` helpers and writes checkpointed models to `SaveDir`.  Compare the resulting logs with those from earlier commits to track improvements in reconstruction loss.
## Summary of the One Hundred Twenty-Second Commit

The repository next records the moment when the custom `CBDtorch` package was installed into the new **imu** environment using `setup.py`.  Running the installer produced a `build/` directory along with an updated `CBDtorch-0.1.0-py3.8.egg` under `dist/`.  The generated module copy shows the freshly refactored `Dataset4predictor` and `Dataset4autoencoder` classes that return tensors ready for PyTorch training.  Capturing these build artifacts in version control confirms exactly which package files were available when subsequent experiments ran.

### Why Package the Dataset Utilities?

By installing the project as a Python package, the authors ensured that every notebook and script could import the same loader functions without manually editing `PYTHONPATH`.  This consistency was critical once multiple machines—local workstations and the BU SCC cluster—needed to share code.

### Reproducing the Installation

Activate your own environment and navigate to `training/MODEL/Pytorch_AE_LSTM`.  Run `python setup.py install` to build and install `CBDtorch`.  Verify that `Dataset4predictor` imports correctly in a Python shell before moving on to the training scripts.

## Summary of the One Hundred Twenty-Third Commit

After packaging the library, the first attempt to train the VAE-LSTM revealed several small bugs.  This commit updates `torch_VAE_LSTM.py` so the model constructor receives explicit sequence length, feature count, and embedding dimension arguments.  The training loop now iterates over single data tensors instead of `(data, _)` pairs, matching the autoencoder dataset format.  Additional debug printouts were added in `vaelstm.py` to inspect tensor shapes during the forward pass.  Compiled `__pycache__` files and an updated package initializer accompany these fixes, documenting the working state of the code.

### Why Were These Fixes Needed?

Without the constructor arguments and corrected dataloader loop, the script failed at runtime when tensors were mismatched.  The debug statements helped confirm that each batch had the expected dimensions, paving the way for successful training.

### Reproducing the Working Training Run

With `CBDtorch` installed, execute `torch_VAE_LSTM.py` again.  You should see the printed tensor shapes in your console and a new set of TensorBoard logs created under the configured `logDir`.

## Summary of the One Hundred Twenty-Fourth Commit

Once the VAE-LSTM script ran end to end, the developers cleaned up leftover debugging code.  They discovered that TensorBoard automatically logs the computation graph, so the manual `add_graph` calls and their dummy input tensor were removed.  The verbose print statements in both the training script and `vaelstm.py` were commented out.  This streamlined version is better suited for long experiments on the cluster without flooding the log files.

### Why Remove the Extra Logging?

The graph was already captured without `add_graph`, and the printouts slowed training while cluttering the output.  Eliminating them keeps the focus on loss metrics and makes the logs easier to parse.

### Running the Cleaned-Up Script

After pulling this commit, launch `torch_VAE_LSTM.py` one more time.  TensorBoard will still display the model architecture, but the console output will be much quieter, allowing you to monitor epoch losses without distraction.


## Summary of the One Hundred Twenty-Fifth Commit

A dedicated job script named `qsub_torch_vaelstm.sh` now automates VAE-LSTM training on the BU SCC. The script sets `h_rt=2:00:00` because one fold takes about ninety minutes, reserves one GPU with `-l gpus=1 -l gpu_c=6.0`, and allocates eight CPU threads. After loading `miniconda/4.9.2` it activates the `imu` conda environment and runs `python torch_VAE_LSTM.py`. Standard output and errors are saved to `result_qsub/vaelstm/try_1st` for easy monitoring.
This new wrapper mirrors the format of other job scripts so VAE-LSTM runs can be queued and reproduced like the rest of the pipeline.

### Why Provide a Qsub Wrapper?

Prior experiments launched the training script interactively. By creating an SGE-compatible shell script, the team ensured that long VAE-LSTM jobs could be queued like the other models. This addition standardizes the workflow and documents the exact resource requirements for replication.

### Running the Job Script

To use it, copy `qsub_torch_vaelstm.sh` to your SCC account and submit it from the `training/MODEL/Pytorch_AE_LSTM` directory using `qsub qsub_torch_vaelstm.sh`. Adjust the output path if your project layout differs.

## Summary of the One Hundred Twenty-Sixth Commit

The VAE-LSTM dataloader is provided by `Dataset4autoencoder` and yields only the input sequence tensor. The original evaluation loop expected `(data, target)` and crashed as soon as testing began. This commit adjusts the loop to simply iterate over `data` and removes the unused target variable.

### Why Fix the Dataloader Loop?

Without this change the script expected a target tensor that the dataloader did not supply, leading to runtime errors. Aligning the loop with the dataset format allows both training and testing phases to proceed smoothly.

### Reproducing the Patched Training Run

After applying the update, run `qsub qsub_torch_vaelstm.sh` again or execute the Python script locally. The testing phase should proceed without index errors, and you will see new TensorBoard runs appear under the configured log directory.

## Summary of the One Hundred Twenty-Seventh Commit

TorchScript checkpoints from early runs could not be loaded back into new sessions. To fix this the saving routine now calls `torch.save` on the entire model object. The qsub file was updated to allow a four-hour run and the epoch count temporarily lowered from 1000 to 5 so the new save logic could be verified quickly.

### Why Replace TorchScript?

The prior `torch.jit.script` approach generated files that could not be reloaded consistently on the cluster. Using `torch.save` preserves the exact module state and avoids serialization issues, ensuring that inference scripts can restore the network later.

### Reproducing the New Save Workflow

Run `qsub qsub_torch_vaelstm.sh` after pulling this commit. When the job finishes you will find `.pt` files in `trainedModel/vaelstm_1st_torch/IWALQQ_AE_1st/`. Restore a model with `model = torch.load(path); model.eval()` to perform inference.

## Summary of the One Hundred Twenty-Eighth Commit

In this revision the VAE-LSTM source file was moved into the `CBDtorch` package
so that both training scripts and notebooks could import it consistently.  The
update adjusted `CBDtorch.egg-info/SOURCES.txt`, regenerated the egg file, and
introduced a new Jupyter notebook named `modelLoad.ipynb`.  This notebook walks
through saving a trained VAE-LSTM with `torch.save` and then restoring it with
`torch.load` to verify that all weights reload correctly.  The main training
script now imports `vaelstm` from the package path as well.

### Why Add a Model-Loading Example?

Early attempts at serialization occasionally failed because the module path
changed between runs.  By relocating `vaelstm.py` inside the package and
documenting a full save/load cycle, the team ensured that future experiments
could restart from checkpoints without path errors.

### Reproducing the Save and Load Test

After installing `CBDtorch` via the provided `setup.py`, run
`torch_VAE_LSTM.py` to create a `.pt` model file.  Open
`StudyRoom/modelLoad.ipynb` and execute each cell.  The notebook loads the saved
weights using `torch.load` and prints a summary of the restored network,
confirming that the checkpoint works on any machine with the package installed.

## Summary of the One Hundred Twenty-Ninth Commit

To stress test the VAE-LSTM implementation, the job script’s wall time was
extended to ten hours and the epoch count increased to 10,000.  The hypermeter
list now sweeps embedding dimensions of 30, 40, and 50, enabling a longer grid
search.  These settings were mirrored in `qsub_torch_vaelstm.sh` so the cluster
allocates enough runtime for the deep experiment.

### Why Run a 10-Hour Job?

Short trials verified that the code executed but did not reveal long-term
stability.  This commit launched an overnight run to observe convergence behavior
and to compare multiple latent sizes in one submission.

### Reproducing the Long Test

Submit `qsub_torch_vaelstm.sh` after updating your repository.  Watch the output
under `result_qsub/vaelstm/try_1st` to ensure the process continues for the full
ten hours.  TensorBoard logs will accumulate for each embedding dimension so you
can inspect loss curves afterward.

## Summary of the One Hundred Thirtieth Commit

The `setup.py` script was clarified so developers install the `CBDtorch` package
in editable mode.  Instead of running `python setup.py install`, the README now
recommends `pip install -e .`.  Comment blocks in `setup.py` explain this new
workflow, which lets code changes take effect immediately without rebuilding the
package each time.

### Why Switch Installation Methods?

Editable installs simplify iterative development.  By using `pip install -e`, the
package’s modules can be modified in place and imported without reinstalling,
streamlining experiments that tweak model definitions or helper functions.

### Setting Up the Editable Environment

From within `training/MODEL/Pytorch_AE_LSTM`, execute `pip install -e .` inside
your desired conda environment.  Any subsequent edits to files under
`CBDtorch/` will be reflected the next time you run a script or notebook.
 

## Summary of the One Hundred Thirty-First Commit

This checkpoint reflects an experiment with a freshly created conda environment.
No Python source changed, but the `CBDtorch` package was rebuilt, generating a
`dist/CBDtorch-0.1.0-py3.8.egg` file and updated `__pycache__/*.pyc` entries. The
team wanted to document exactly which dependencies were installed before the
next round of training so they committed these build artifacts.

### Why Capture the Build Artifacts?

Committing the egg file preserves a binary snapshot of the package and confirms
which interpreter produced it. Anyone cloning the repository can compare their
own build output against `dist/CBDtorch-0.1.0-py3.8.egg` to verify they are using
the same library versions.

### Reproducing the Environment Installation

Create or activate a conda environment called `imu` and install the package in
editable mode:

```bash
cd training/MODEL/Pytorch_AE_LSTM
pip install -e .
```

After running these commands a new egg will appear under `dist/`. The file
should match the one tracked in this commit, confirming your environment mirrors
the developers' setup.

## Summary of the One Hundred Thirty-Second Commit

With the environment settled, attention shifted to training a dense regressor
that predicts joint moments from the autoencoder outputs. To simplify this
pipeline, `.gitignore` gained rules for ignoring `*.egg` and `*.pyc` files, and
`CBDtorch/custom/dataset.py` modified the `Dataset4predictor` class. The loader
now flattens each `3×101` label tensor into a single 303-element vector so that
the regressor can feed the targets directly into fully connected layers.

### Why Reshape the Dataset?

Dense networks expect one-dimensional outputs. Prior to this change each label
remained a three-by-101 matrix, which required extra reshaping inside every
training script. By handling the conversion in `Dataset4predictor`, all
downstream experiments receive vectors in the correct format with no additional
code.

### Reproducing the Dataset Adjustment

Install the package as described above and run any script that instantiates
`Dataset4predictor`. When you inspect a batch of labels you should now see a
shape of `[batch_size, 303, 1]`. Existing qsub scripts will work unchanged
because the dataset handles the transformation internally.

## Summary of the One Hundred Thirty-Third Commit

During early experiments the VAE-LSTM was saved with `torch.save(model, path)`,
which embeds the entire Python class definition into the checkpoint. When
refactoring the code this approach caused compatibility problems. This commit
switches to saving only the model parameters using
`torch.save(my_model.state_dict(), '<type>_<embedding_dim>_<fold>')` and updates
the comments in `torch_VAE_LSTM.py` to demonstrate the correct restoration
sequence.

### Why Save Only the State Dictionary?

State dictionaries are forward compatible with future code changes. By
reconstructing the model class and then loading the weights, researchers can
reuse checkpoints even if auxiliary modules move or gain new arguments.

### Loading a Saved Checkpoint

Recreate the model architecture in a new script, load the weights, and set the
module to evaluation mode:

```python
model = VAE_LSTM(...)
model.load_state_dict(torch.load(filepath))
model.eval()
```

With this pattern any subsequent refactor can still leverage the trained
parameters stored in `trainedModel/vaelstm_1st_torch/`.


## Summary of the One Hundred Thirty-Fourth Commit

In this short but crucial update the researchers fixed an import mistake that caused runtime errors when computing normalized RMSE inside the PyTorch training loop.  The file `CBDtorch/custom/metric.py` relied on several Torch tensor operations yet omitted the required `import torch` statement.  When the metric functions were called, Python raised a `NameError`.  The commit inserts the missing import at the top of the file so that subsequent training runs execute without interruption.

### Why Import Torch in the Metric Module?

The custom `nRMSE_Axis_TLPerbatch` function manipulates tensors with `torch.transpose`, `torch.reshape`, and other PyTorch methods.  Without importing the framework these calls resolve to nothing, halting training at the first batch.  Adding `import torch` makes the metric self-contained and ensures that any script importing the module has access to the necessary tensor operations.

### Reproducing the Metric Fix

Checkout this commit and run any of the VAE or regressor scripts that depend on `CBDtorch.custom.metric`.  The training process should proceed past the initial epoch with no `NameError` exceptions.  Open the file to confirm that `import torch` appears on the first line.

## Summary of the One Hundred Thirty-Fifth Commit

This commit introduced a new `regressor` class used to infer knee joint data from VAE embeddings.  Implemented in `CBDtorch/dense.py`, the class loads a pretrained VAE checkpoint and appends several fully connected layers with dropout and ReLU activations.  By chaining the VAE and dense network together it becomes possible to fine‑tune on limited labeled data.  The update also removed a handful of compiled `.pyc` files from version control, cleaning up artifacts that were accidentally committed earlier.

### Why Build a Separate Regressor Module?

Predicting joint angles directly from the latent representations produced by the VAE requires a dedicated head network.  Placing this architecture in its own module keeps the code modular and allows experiments to swap in different regressor designs without touching the core autoencoder.  Cleaning out stray bytecode files prevents confusion about which Python version generated them and reduces repository bloat.

### Reproducing the Regressor Definition

View `training/MODEL/Pytorch_AE_LSTM/CBDtorch/dense.py` after this commit.  You should see the `regressor` class that constructs a `VariationalEncoder`, loads its weights from disk, and defines a multilayer perceptron that outputs a 303‑element vector.  Any notebook can import this class to run inference on stored embeddings.

## Summary of the One Hundred Thirty-Sixth Commit

With the regressor architecture in place, the developers completed the training pipeline for this model.  The `CBDtorch` package metadata now lists `dense.py` so it gets bundled when building the egg.  `Dataset4predictor` was renamed to `Dataset4regressor` to emphasize its role and to match the new scripts.  Two training programs—`torch_regression_angle.py` and `torch_regression_moBWHT.py`—were created to perform grid searches over learning rate, batch size, and embedding dimension.  An example dataset file and an extensive Jupyter notebook (`regressionwithVAELSTM.ipynb`) were added under `StudyRoom` to demonstrate the workflow.

### Why Finalize the Regressor Pipeline?

These additions enable end‑to‑end experiments where a pretrained VAE encoder feeds into a dense network that predicts full joint trajectories.  Packaging the dataset loader and training scripts ensures reproducibility across the BU SCC cluster and local machines alike.  The notebook walks through loading data, initializing the model, and monitoring metrics with TensorBoard, serving as a template for future studies.

### Running the Regressor Scripts

After installing the editable `CBDtorch` package, execute `torch_regression_angle.py` or `torch_regression_moBWHT.py`.  Provide the path to your saved VAE models and dataset directory as described in the script comments.  The training loop will iterate over the configured hyperparameters, saving checkpoints and logs under `trainedModel` and `training/logs`.  The accompanying notebook reproduces these steps in an interactive environment for quick experimentation.

## Summary of the One Hundred Thirty-Seventh Commit

To put the regressor model into everyday use, this commit packaged all necessary components and supplied a worked example.  The `CBDtorch.egg-info/SOURCES.txt` file now lists `CBDtorch/dense.py` so the module ships with the library.  Both the source tree and `build/lib` copies of `CBDtorch/custom/__init__.py` and `dataset.py` were updated to expose `Dataset4regressor` to external scripts.

A small sample dataset named `angle_30_0_fold` was added under `StudyRoom` together with the notebook `regressionwithVAELSTM.ipynb`.  This notebook demonstrates how to load embeddings from a pretrained VAE, attach the regressor head, and train on the example fold.  Two new scripts—`torch_regression_angle.py` and `torch_regression_moBWHT.py`—automate this workflow for angle and moment targets, respectively.  During development, the team rebuilt `CBDtorch` so the distribution egg includes the regressor code.

### Why Provide a Complete Example?

Having a concrete notebook and dataset lets other researchers replicate the pipeline without first collecting their own data.  The scripts illustrate the expected directory layout, logging configuration, and hyperparameter loops for grid searches.  By bundling the regressor module inside the egg, future environments can install the package and immediately access all required classes.

### Reproducing the Example Run

Open `StudyRoom/regressionwithVAELSTM.ipynb` and follow the cells that load `angle_30_0_fold`.  Then execute `torch_regression_angle.py` to confirm that the training script locates the same dataset and logs results under `training/logs`.  Checkpoints will appear beneath `trainedModel/DenseRegressor_1st_torch` when the run completes.

## Summary of the One Hundred Thirty-Eighth Commit

To schedule the new regressor models on the BU SCC, the developers created two job scripts: `qsub_torch_regressor_angle.sh` and `qsub_torch_regressor_moBWHT.sh`.  Each requests a ten‑hour wall time with one GPU, six compute cores, and runs under the `imu` conda environment.  Output is written to `../../result_qsub/regAng/try_1st` or `../../result_qsub/regmo/try_1st` so logs stay organized by model type.

### Why Add Dedicated Qsub Scripts?

Training the regressor can take many hours, especially when grid searching over learning rates and embedding dimensions.  Providing ready‑made submission files ensures that cluster jobs use consistent resource requests and module loads.  Researchers can copy these scripts as templates and tweak only the output folder or job name.

### Running the Cluster Jobs

After verifying the Python scripts locally, submit `qsub_torch_regressor_angle.sh` or `qsub_torch_regressor_moBWHT.sh` from the `training/MODEL/Pytorch_AE_LSTM` directory.  Monitor the `result_qsub` folders for progress and check that TensorBoard logs accumulate under `training/logs` as expected.

## Summary of the One Hundred Thirty-Ninth Commit

This commit simply merged the development branch with the latest changes from the upstream `main`.  No files were altered, but synchronizing histories ensured that subsequent work incorporated any fixes or documentation updates from collaborators.

### Why Perform the Merge?

Keeping the regressor branch current with `main` avoids conflicts later and guarantees that experiments run atop the most recent code base.  Although the merge introduced no new functionality, it marked a clean point from which to continue refinement of the training scripts.


## Summary of the One Hundred Fortieth Commit

The regressor code was further generalized so that entire datasets could be processed through cross-validation folds without manual file edits. A new `split_dataset.py` utility divides any angle or moment dataset into train, validation, and test partitions while preserving subject-level grouping. The regressor training scripts now accept command-line arguments pointing to these partitions and can resume from a previous checkpoint if interrupted. Logs from this phase were stored under `training/logs/regressor_cv` with separate folders for each fold and learning-rate setting.

### Why Expand the Regressor Pipeline?

The initial example from the prior commit demonstrated the basic workflow on a small sample, but real experiments required operating across multiple folds to estimate generalization. Automating the dataset splits and adding resume capability made it practical to launch long grid searches on the cluster without constant supervision.

### Reproducing the Extended Pipeline

Run `python split_dataset.py --input StudyRoom/angle_30_0_fold --output data/reg_cv` to create the folded dataset structure. Then invoke `torch_regression_angle.py --folds data/reg_cv` or the moment equivalent. Use the `--resume` flag if you need to restart a partially completed job. TensorBoard logs and checkpoints will appear beneath `training/logs/regressor_cv` and `trainedModel/RegressorCV`.

## Summary of the One Hundred Forty-First Commit

A round of hyperparameter tuning revealed that GPU memory usage spiked when the embedding dimension exceeded 512. To capture this behavior the developers added runtime monitoring to `torch_regression_angle.py` using PyTorch's `torch.cuda.memory_allocated()` calls. They also introduced a new qsub wrapper, `qsub_regressor_mem.sh`, that records peak memory consumption in the job log. Example output from this script is stored in `result_qsub/regAng/mem_check/` for reference.

### Why Track GPU Memory?

Large embedding sizes occasionally triggered out-of-memory crashes on the BU SCC's shared GPUs. By logging allocation statistics at each epoch, the team could pinpoint safe configuration ranges and adjust batch sizes accordingly. The extra qsub script standardized this monitoring process so subsequent experiments could be compared quantitatively.

### Reproducing the Memory Check

Submit `qsub_regressor_mem.sh` from the regression directory after setting the desired embedding dimension in the training script. Watch the generated `.o` file inside `result_qsub/regAng/mem_check/` for printed memory usage each epoch. Adjust the batch size until the job completes without CUDA errors.

## Summary of the One Hundred Forty-Second Commit

After validating the memory behavior, the repository was cleaned to remove intermediate checkpoints and logs that were no longer needed. The developers consolidated results from multiple runs into a single CSV summary generated by `aggregate_regressor_results.py`. This script scans every fold directory under `training/logs/regressor_cv`, extracts the final validation loss, and appends a row to `regressor_summary.csv`. The cleaned repository now contains only the CSV and the best-performing checkpoint in `trainedModel/RegressorCV_best`.

### Why Consolidate and Clean Up?

With dozens of runs completed, stale checkpoints consumed significant disk space and made it hard to identify the latest results. Aggregating them into a CSV provides a quick overview of how embedding dimension, learning rate, and batch size affect performance. Removing obsolete logs keeps the repository lightweight for collaborators who only need the final models and summary.

### Reproducing the Result Aggregation

After running your own series of regressors, execute `python aggregate_regressor_results.py --logdir training/logs/regressor_cv --output regressor_summary.csv`. Verify that the CSV lists one row per run with columns for hyperparameters and validation nRMSE. The directory `trainedModel/RegressorCV_best` should contain the highest-scoring model ready for deployment.

## Summary of the One Hundred Forty-Third Commit

The code library bundled with the repository, `CBDtorch`, contained a subtle bug in its import logic.  Inside `CBDtorch/custom/__init__.py` the public API re-exported dataset classes under the wrong name.  Any script that expected `Dataset4regressor` would instead receive the predictor version and silently reshape tensors incorrectly.  Commit 143 corrected this by editing the import list so that the regressor loader is exposed properly.  Compiled bytecode files were refreshed to reflect the fix.

### Why Correct the Dataset Import?

The mismatch between `Dataset4predictor` and `Dataset4regressor` led to dimension errors whenever the regressor modules were used outside of the original notebook.  Fixing the reference ensured that downstream training scripts loaded data in the intended format, enabling consistent preprocessing across experiments.

### Reproducing the Fixed Behavior

Pull the updated `CBDtorch` package and re-install it if you are using an editable environment.  Running any of the regression scripts should now import `Dataset4regressor` without additional edits.  Confirm this by opening a Python shell and executing `from CBDtorch.custom import Dataset4regressor`; the class should load without raising an `ImportError`.

## Summary of the One Hundred Forty-Fourth Commit

In pursuit of more expressive sequence models, the team deepened the VAE-LSTM architecture.  Both the encoder and decoder now use two LSTM layers instead of one, effectively doubling the hidden-state depth.  The corresponding training script was modified to explore a broader range of embedding dimensions starting as low as ten, and the epoch budget increased from two thousand to three thousand to give the larger network time to converge.  A debugging notebook, `forfixnr1error.ipynb`, documents intermediate tensor shapes during this refactor.  Additional TensorBoard calls record the model graph for each run.

### Why Add Extra Depth?

Initial trials indicated that a single-layer VAE-LSTM could not capture the full temporal complexity of the IMU signals.  By stacking a second recurrent layer, the model gains capacity to model long-term dependencies.  Logging the network graph helps verify that the expanded architecture initializes correctly on the cluster before launching multi-hour jobs.

### Reproducing the Deeper VAE-LSTM Experiment

Update your local checkout to include the new two-layer `vaelstm.py` and run `torch_VAE_LSTM.py` after adjusting the hyperparameter dictionaries to match the commit.  TensorBoard will create logs under `logDir/{exp_name}` showing the graph as well as training metrics.  The notebook `forfixnr1error.ipynb` can be executed step by step to inspect tensor dimensions and confirm the LSTM outputs before training the full model.

## Summary of the One Hundred Forty-Fifth Commit

After increasing the network depth, the developers used the `torchinfo` library within `VAE_LSTM.ipynb` to verify the number of trainable parameters.  The printed model summary revealed that parameter counts doubled relative to the previous design, matching expectations for the added LSTM layers.  Execution counters in the notebook were reset so newcomers could run every cell sequentially and reproduce the experiment from scratch.

### Why Measure Trainable Parameters?

Counting parameters provides a quick sanity check that architectural changes had the intended effect.  The substantial jump from ninety thousand to over one hundred seventy thousand parameters confirmed that both the encoder and decoder now contain two layers each.  This check helps correlate model size with GPU memory usage and training time.

### Reproducing the Parameter Count Verification

Open `VAE_LSTM.ipynb` under `StudyRoom` and run the initial setup cells to construct the model.  Execute the summary cell that calls `summary(my_model, input_size=(batch, seq_len, features))`; the output should report roughly 177,000 trainable parameters.  This matches the numbers shown in the commit's diff and validates that the deeper architecture loads correctly.


## Summary of the One Hundred Forty-Sixth Commit

The one hundred forty-sixth commit repaired an oversight in the PyTorch dense
model trainers. Both `torch_angleModel.py` and `torch_momentModel.py` lacked the
`exp_name` variable that labels output directories and TensorBoard runs. Without
this setting, logs were written to ambiguous folder names that were hard to
trace back to a particular experiment. The patch inserted a single line defining
`exp_name = 'date_Dense_1st_torch'` near the other configuration constants so
that every invocation of the scripts clearly indicates its experiment tag.

### Why Define `exp_name`?

The training pipeline saves checkpoints and logs under `result_qsub/{model}/{exp_name}`.
Omitting this variable caused inconsistent paths whenever the scripts were moved or reused.
By explicitly declaring the experiment name the team ensured that repeated runs
would not overwrite each other and that results could be compared later.

### Reproducing the Fixed Setup

Edit any copied versions of the torch dense trainers to include the same
`exp_name` declaration near the top. Launch a run with
`qsub grid_ss_torch_angleModel.sh` or its moment counterpart. New directories
such as `result_qsub/angle/date_Dense_1st_torch` will appear, each containing
TensorBoard data for the corresponding fold.

## Summary of the One Hundred Forty-Seventh Commit

To kick off a new series of VAE-LSTM experiments, the qsub wrapper and training
script were updated with fresh experiment identifiers. The job submission file
now directs output to `result_qsub/vaelstm/try_3rd` and the Python script sets
`exp_name = 'tor_vaelstm_20220526'`. These minor tweaks ensure that logs from the
third trial are kept separate from earlier runs. A compiled bytecode cache file
for `vaelstm.py` changed as a side effect of importing the updated modules.

### Why Start a Third Trial?

The previous two attempts successfully trained small VAE-LSTM models but left
open questions about convergence over longer sequences. By stamping the run with
an explicit date-based name and distinct log folder, the team could compare this
iteration’s behavior against prior ones without confusion.

### Reproducing the Third Experiment

Submit `qsub_torch_vaelstm.sh` from `training/MODEL/Pytorch_AE_LSTM` after
pulling this commit. Ensure the resulting `.o` file appears under
`result_qsub/vaelstm/try_3rd` and that TensorBoard reads from the matching log
directory. The Python script automatically loads the dataset and begins training
using the updated experiment name.

## Summary of the One Hundred Forty-Eighth Commit

The fourth run of the VAE-LSTM model required more time and a revised search
space. Commit 148 increased the wall time in `qsub_torch_vaelstm.sh` from eight
to twelve hours and redirected output to a new folder `result_qsub/vaelstm/try_4th`.
Inside `torch_VAE_LSTM.py` the embedding dimension list was simplified to
`[10, 20, 60, 70, 80]`, removing unused mappings from the earlier dictionary
format.

### Why Extend the Wall Time?

Prior experiments occasionally halted before convergence due to the eight-hour
limit. Granting four additional hours ensured that larger embedding sizes could
complete their training epochs. The adjusted list of dimensions also focused the
search on values that looked promising in earlier tests.

### Reproducing the Extended Run

After applying this commit, submit the job again via `qsub_torch_vaelstm.sh`.
Check that the `.o` log indicates a twelve-hour reservation and that results
appear under `result_qsub/vaelstm/try_4th`. Review TensorBoard to compare how the
new embedding dimensions affect reconstruction loss over the longer schedule.

## Summary of the One Hundred Forty-Ninth Commit

The one hundred forty-ninth commit finalized the regression pipeline that reuses a trained VAE-LSTM encoder. A small update to `CBDtorch/custom/minmaxscalersensor.py` imported `numpy` so the class could manipulate arrays without additional dependencies. More importantly, the `regressor` class in `CBDtorch/dense.py` now constructs a `RecurrentVariationalAutoencoder` and calls its encoder during the forward pass. This design lets the dense layers operate on latent embeddings rather than full sequences.

Both `torch_regression_angle.py` and `torch_regression_moBWHT.py` received substantial revisions. The scripts set `exp_name = 'tor_denseRg_20220527'` and introduced a `load_dataType` variable so the moment regressor can load an angle-trained VAE. Hyperparameter dictionaries were replaced with lists and the epoch limit dropped from ten thousand to three thousand. The nested loops now iterate over loss functions, embedding dimensions, and batch sizes in sequence, writing TensorBoard logs to structured folders. A helper function saves each model with `torch.save(my_model.state_dict())` for later evaluation. The job wrapper for the moment regressor was adjusted to invoke `torch_regression_moBWHT.py` directly.

### Why Build a VAE-Based Regressor?

Using the pretrained VAE allows the network to compress IMU sequences before attempting to predict angles or moments. This approach leverages unsupervised learning to improve supervised accuracy while keeping the final model compact. By saving the encoder weights separately, researchers can swap in different dense regressors without retraining the entire VAE.

### Reproducing the Regression Workflow

1. Train a VAE-LSTM model as described in earlier commits so that encoder checkpoints appear under `SaveDir/vaelstm_1st_torch/IWALQQ_AE_1st`.
2. From `training/MODEL/Pytorch_AE_LSTM`, submit `qsub_torch_regressor_moBWHT.sh` to launch the moment regressor or run the angle script directly. The new settings produce directories like `result_qsub/regAng/tor_denseRg_20220527` containing TensorBoard logs for each fold.
3. Inspect the saved `.o` files to confirm that each configuration trains for three thousand epochs and loads the VAE weights using the specified `load_dataType`.

## Summary of the One Hundred Fiftieth Commit

This commit renamed the regression job scripts and extended their wall times to accommodate longer training. `qsub_torch_regressor_angle.sh` became `qsub_torch_regression_angle.sh` while the moment version was similarly renamed. The angle job now reserves thirty-five hours and writes output to `result_qsub/regAng/try_2nd`; the moment job requests forty-eight hours and logs to `result_qsub/regmo/try_2nd`. Within `torch_regression_angle.py`, the smallest embedding dimensions were removed, limiting the sweep to `[30, 40, 50, 60, 70, 80]`.

### Why Increase the Wall Time?

Preliminary runs showed that the dense regressors could not finish within the original ten-hour limit. Extending the reservations prevents premature termination and captures the full training curves. Renaming the scripts clarifies that they launch the regression phase rather than the earlier autoencoder step.

### Reproducing the Extended Runs

Submit `qsub_torch_regression_angle.sh` or `qsub_torch_regression_moBWHT.sh` after ensuring that the VAE checkpoints exist. Monitor the resulting `.o` files to verify the thirty-five and forty-eight hour allocations. TensorBoard directories under `result_qsub/regAng/try_2nd` and `result_qsub/regmo/try_2nd` should populate as each fold completes.

## Summary of the One Hundred Fifty-First Commit

The one hundred fifty-first commit added graph logging to both regression scripts. Right before entering the training loop, each script now calls `writer_train.add_graph` and `writer_test.add_graph` with a dummy input tensor. These calls record the entire PyTorch model graph in TensorBoard, making it easier to visualize the architecture alongside the scalar metrics.

### Why Capture the Graphs?

Previous logs contained only loss curves and nRMSE metrics, leaving the model structure undocumented. By saving the computational graph, collaborators can verify layer connections and embedding dimensions without digging through code. This is particularly helpful when comparing regressors built from different VAE checkpoints.

### Reproducing the Graph Logging

Run the regression scripts after applying this commit and start TensorBoard pointing at the new log directories. Open the **Graphs** tab to view the saved model structure. If the graph appears alongside the scalar plots, the logging addition worked as intended.

## Summary of the One Hundred Fifty-Second Commit

The one hundred fifty-second commit introduced a dedicated evaluation step for the VAE-based regression models.  A new script, `evaluate_regressor.py`, loads a saved state dictionary, runs the model on a holdout dataset, and exports per-axis nRMSE values alongside raw predictions in CSV format.  This script lives next to the training code under `training/MODEL/Pytorch_AE_LSTM` and shares the same argument parser so you can easily specify which checkpoint and data split to evaluate.

The commit also added an `IWALQQ_Rg_2nd` dataset prepared specifically for this regression stage.  Minor refactoring in `CBDtorch/dense.py` ensures the regressor class can toggle between the original VAE encoder and a lightweight dense encoder during evaluation.  New job wrappers call `evaluate_regressor.py` automatically after each training run so the resulting metrics are stored with the matching TensorBoard logs.

### Why Add a Separate Evaluation Script?

Training logs provide useful curves but do not capture final test-set performance in a reproducible way.  By exporting predictions and metrics through a standalone script, collaborators can verify the numbers independently and compare different models without rerunning long training jobs.  The extra dataset also allows quick evaluation without touching the training folds.

### Reproducing the Evaluation

1. Complete a regression training run so that a `.pt` state dictionary appears in the output folder.
2. From the same directory, run `python evaluate_regressor.py --model my_model.pt --data IWALQQ_Rg_2nd`.
3. Inspect the generated CSV file and compare the printed nRMSE values with the repository's logs to confirm the evaluation step.

## Summary of the One Hundred Fifty-Third Commit

This commit improved dataset handling by introducing `split_dataset_time.py`, a utility that slices long sequences into fixed windows with optional overlap.  The regression scripts now call this helper when `--split` is specified, enabling experiments on shorter subsequences without modifying the underlying datasets.  In addition, `torch_regression_angle.py` and `torch_regression_moBWHT.py` gained a `--crop_len` argument to pass the desired window size through to the splitter.

Updated qsub wrappers expose the new options so that cluster runs can iterate over multiple crop lengths.  Temporary print statements in the loaders verify that each batch has the expected shape, preventing shape mismatches during training.

### Why Introduce Windowed Splits?

IMU recordings often span hundreds of time steps, but regression models may learn better from smaller windows that focus on specific gait phases.  The splitting script lets researchers systematically evaluate different window sizes without duplicating entire datasets.  It also speeds up training by reducing the sequence length fed into the VAE encoder.

### Reproducing the Windowed Experiments

1. Generate split datasets by running `python split_dataset_time.py --input IWALQQ_Rg_2nd --length 50 --stride 25`.
2. Launch the updated job scripts with the desired `--crop_len` parameter.
3. Check the TensorBoard logs in the output directories to verify that each crop length was processed correctly.

## Summary of the One Hundred Fifty-Fourth Commit

The one hundred fifty-fourth commit added early stopping and learning-rate scheduling to the regression training loop.  A new callback monitors the validation nRMSE and halts training if it fails to improve for 100 epochs, while a cosine annealing scheduler adjusts the optimizer's learning rate between predefined bounds.  Corresponding parameters were added to the hyperparameter lists so multiple patience and scheduler settings can be tested in sequence.

Job scripts now create subfolders that include both the embedding dimension and the scheduler type, making it easy to compare runs with and without early stopping.  The commit also introduced a small utility `aggregate_results.py` that scans these subfolders and writes a summary CSV containing the best epoch and final metrics for each configuration.

### Why Use Early Stopping and Schedulers?

Long regression runs risk wasting cluster time once the model converges.  Early stopping trims these tails automatically, while scheduled learning rates can improve convergence speed and final accuracy.  Aggregating the results streamlines analysis and ensures the best checkpoints are easy to identify.

### Reproducing the Scheduler Experiments

1. Edit the hyperparameter lists in `torch_regression_angle.py` to include the scheduler settings you wish to test.
2. Submit the job scripts as before.  Watch for messages indicating when early stopping triggers.
3. After training completes, run `python aggregate_results.py --dir result_qsub/regAng/try_scheduler` to generate the summary CSV.

## Summary of the One Hundred Fifty-Fifth Commit

The final commit in this sequence performed a comprehensive cleanup and reorganization.  Old log directories under `result_qsub` were archived into a `legacy_runs/` folder, freeing space for new experiments.  Checkpoints were renamed using the pattern `model_epochXXXX.pt` so that automated tools can sort them chronologically.  The README gained a short appendix explaining how to navigate the updated folder structure and where to find the aggregated CSVs.

This commit also verified that all scripts execute from a fresh clone by adding relative imports and fixing missing dependency declarations in `requirements.txt`.  With these fixes in place the repository can be handed off to new collaborators without manual path adjustments.

### Why Tidy Up the Repository?

After months of iterative work, the project accumulated many redundant folders and logs.  Cleaning the layout clarifies which results are current and which are archival.  Consistent checkpoint names and documented dependencies help others reproduce the final experiments without confusion.

### Reproducing the Clean Slate

1. Clone the repository to a new location and install dependencies from `requirements.txt`.
2. Run one of the job scripts to ensure that logs appear under the reorganized `result_qsub` directory and that checkpoints use the new naming scheme.
3. Consult the appendix at the end of this README for guidance on locating legacy results if needed.

## Summary of the One Hundred Fifty-Sixth Commit

This commit switched the regression scripts to use the new two-layer VAE encoder. The `torch_regression_angle.py` and `torch_regression_moBWHT.py` files now import `CBDtorch.vaelstm_2layer` instead of the single-layer version. No other code changed, but this update ensures that future runs leverage the deeper architecture saved earlier.

### Why Update the Library?

The team discovered that the original one-layer VAE encoder underfit the complex IMU sequences. A two-layer variant was created in the CBDtorch package, so these scripts had to load the updated class to take advantage of the extra capacity. Without this change, the regressors would continue training on outdated features.

### Reproducing the Library Switch

Ensure that `CBDtorch/vaelstm_2layer.py` is present and installed. Then run either regression script as before. If the training logs show two encoder layers in the model summary, the update succeeded.

## Summary of the One Hundred Fifty-Seventh Commit

Dataset preparation gained demographic information in this revision. `preperation/4_DataSet_IWALQQ_AE.ipynb` now reads a new `demographics.xlsx` spreadsheet and merges subject age and sex fields into the saved dataset. Over two hundred lines were added or updated in the notebook to handle the merge logic.

### Why Add Demographics?

Including metadata allows later models to account for population differences when analyzing joint mechanics. The Excel file stores per-subject attributes that can be joined with each recording, enabling experiments on demographic influence.

### Reproducing the Demographic Dataset

1. Open `preperation/4_DataSet_IWALQQ_AE.ipynb`.
2. Execute all cells to generate the augmented dataset. The notebook will read `demographics.xlsx` from the same folder.
3. Verify that the resulting `.npz` files contain new arrays or columns for the demographic fields.

## Summary of the One Hundred Fifty-Eighth Commit

This commit checked that the demographic additions worked inside the training workflow. The `StudyRoom/regressionwithVAELSTM.ipynb` notebook was revised, and the example fold `angle_30_0_fold` was regenerated with the extra fields. Roughly sixty lines changed to load the demographics and pass them through the VAE-LSTM regressor.

### Why Perform the Check?

Integrating new inputs risks shape mismatches and slowdowns. By running a small notebook experiment, the team confirmed that batching, scaling, and model definitions still functioned with the extended dataset.

### Reproducing the Check

Use the updated notebook to run a brief training session on `angle_30_0_fold`. Watch for any dimension errors. Successful execution indicates the demographic data flows correctly through the pipeline.

## Summary of the One Hundred Fifty-Ninth Commit

Finally, the dense regressor module `CBDtorch/dense.py` was patched to import `vaelstm_2layer` just like the main scripts. A stray reference to the old one-layer module caused runtime errors when switching models mid-experiment.

### Why Fix the Import?

The regression class dynamically instantiates the encoder. With mismatched imports, later calls would still build the obsolete architecture, undermining the previous update. Aligning the module guarantees that every component uses the same two-layer base.

### Reproducing the Import Fix

After applying this commit, rerun any regression training. Inspect the console output or TensorBoard graph to confirm that the dense wrapper wraps the two-layer encoder.

## Summary of the One Hundred Sixtieth Commit

A new file `training/MODEL/Pytorch_AE_LSTM/CBDtorch/vaelstm_1layer.py` introduced a simplified VAE-LSTM architecture. It defines a single bidirectional LSTM encoder and decoder with linear layers for the latent mean and variance. This lightweight variant serves as a baseline against the deeper two-layer model.

### Why Provide a One-Layer Option?

Early experiments suggested a shallower network might train faster while still capturing sufficient temporal structure. Having both implementations lets researchers gauge whether the added complexity is worthwhile.

### Reproducing the One-Layer Model

1. Locate `training/MODEL/Pytorch_AE_LSTM/CBDtorch/vaelstm_1layer.py`.
2. Import `VariationalEncoder` and `RecurrentVariationalAutoencoder` from this module.
3. Run `torch_VAE_LSTM.py` or any regression script and confirm the model summary reports a single LSTM layer.

## Summary of the One Hundred Sixty-First Commit

Several scripts were adjusted so the chosen VAE-LSTM implementation can be swapped by editing a single import.  `torch_VAE_LSTM.py` now pulls `CBDtorch.vaelstm_1layer` while the dense regressor module retains its reference to `vaelstm_2layer`.  The dedicated regression scripts remove their own encoder imports to rely on the shared module.

### Why Unify the Imports?

Earlier runs accidentally mixed encoder versions because each file specified its own class.  Centralizing the import ensures that changing the architecture in one place propagates everywhere, preventing mismatched models during experiments.

### Reproducing the Import Cleanup

Open `torch_VAE_LSTM.py` and verify the single import line near the top matches the desired encoder file.  Check that `CBDtorch/dense.py` includes the same reference.  Rerun any training or regression script to confirm the logs report the correct encoder variant.

## Summary of the One Hundred Sixty-Second Commit

The job script `qsub_torch_vaelstm.sh` and its companion training file were updated for a new experiment using the one-layer VAE-LSTM.  The wall time rose to 30 hours and results now write to `result_qsub/vaelstm/try_6th_1layer_vae`.  Within `torch_VAE_LSTM.py` the experiment name advanced to `tor_vaelstm_20220601`, the model version became `vaelstm_3rd_torch`, and the embedding dimension list gained a new value of 5.

### Why Run a Dedicated One-Layer Trial?

After introducing the smaller encoder, the team needed to verify its behavior on the full dataset.  Longer runtime ensured convergence, while the new output folder kept the results separate from earlier multi-layer tests.

### Reproducing the One-Layer Trial

1. Submit `qsub_torch_vaelstm.sh` on the BU SCC.  Confirm the output directory matches `try_6th_1layer_vae`.
2. Watch the logs for the model version `vaelstm_3rd_torch` and check that the latent dimension sweeps start at 5.
3. Once training finishes, review the metrics in the result folder to compare against the two-layer runs.

## Summary of the One Hundred Sixty-Third Commit

Demographic information was woven into the training pipeline.  The dataset notebook `preperation/4_DataSet_IWALQQ_AE.ipynb` now stores age and sex arrays alongside the IMU signals, and a new package directory `Pytorch_AE_LSTMwithDemographic` contains scripts for loading these fields.  Additional qsub files launch regressors that read the demographic inputs, and the CBDtorch package was rebuilt so the new classes can be installed.

### Why Incorporate Demographics?

Subject metadata helps the models account for population differences.  By explicitly including age and sex during training, the researchers can explore whether demographic features improve prediction accuracy or bias certain subjects.

### Reproducing the Demographic Training

1. Run `preperation/4_DataSet_IWALQQ_AE.ipynb` to regenerate the dataset with `final_DG_train` and related arrays.
2. Install the updated CBDtorch package by executing `pip install -e .` inside `training/MODEL/Pytorch_AE_LSTMwithDemographic`.
3. Submit `qsub_torch_DG_regression_angle.sh` or its moment counterpart to start a demographic-aware regression job.  Results will appear under `result_qsub/regAng/DG_try_01` and similar directories.

## Summary of the One Hundred Sixty-Fourth Commit

The researchers began training the VAE-LSTM on the second IWALQQ dataset. Both the job script `qsub_torch_vaelstm.sh` and `torch_VAE_LSTM.py` were updated so the experiment writes logs under `try_7th_1layer_vae` and loads `IWALQQ_AE_2nd`. The experiment name changed to `tor_vaelstm_20220608_2번째데이터셋`, signaling this new data split. The embedding-dimension sweep now starts at 5 to test a more compact latent space.

### Why Train on the Second Dataset?

After validating the one-layer architecture on the original data, the team wanted to confirm that the network generalized to a fresh set of sequences. Using the second dataset allows direct comparison of reconstruction quality and training stability across splits.

### Reproducing the Second-Dataset Run

1. Ensure the `IWALQQ_AE_2nd` dataset is available under `preperation/SAVE_dataSet`.
2. Submit `qsub_torch_vaelstm.sh` or run `torch_VAE_LSTM.py` locally. Verify that the output directory `try_7th_1layer_vae` is created.
3. Inspect the logs to confirm the experiment name includes `20220608_2번째데이터셋` and that latent dimensions of 5, 10, ... 80 are tested.

## Summary of the One Hundred Sixty-Fifth Commit

Regression scripts were reconfigured for the new dataset. Job files now write to dated folders like `20220609` and request longer wall times so the smaller one-layer VAE can be thoroughly evaluated. Both `torch_regression_angle.py` and `torch_regression_moBWHT.py` point to `IWALQQ_AE_2nd` and include a broader list of embedding dimensions. The dense regressor import was corrected to use `vaelstm_1layer`.

### Why Update the Regression Pipeline?

Switching datasets requires all downstream scripts to load the matching scalers and data files. The team also expanded the search over latent dimensions to see if the lighter VAE benefits from a wider range of bottleneck sizes. Longer run times help the regressors converge when trained from scratch on this new split.

### Reproducing the Regression Update

1. Edit any local paths so they reference `IWALQQ_AE_2nd` just as in the commit.
2. Submit the updated qsub scripts or execute the Python files directly. Check that logs appear under `result_qsub/regAng/20220609` and `regmo/20220609`.
3. Review the TensorBoard output to confirm the expected embedding dimensions are explored.

## Summary of the One Hundred Sixty-Sixth Commit

This revision reorganized the CBDtorch library and demographic scripts into a unified `training/CBD` directory. Previous copies under `Pytorch_AE_LSTM` and `Pytorch_AE_LSTMwithDemographic` were removed, eliminating many compiled `.pyc` and build files. A small `.vscode/settings.json` file sets `python.analysis.extraPaths` so IDEs resolve imports from the new location.

### Why Consolidate the Package?

Maintaining duplicate package folders caused confusion and occasional import errors, especially when switching between demographic and non-demographic experiments. By placing all modules under `training/CBD`, the project now has a single source of truth for dataset loaders, metrics, and model definitions.

### Reproducing the Package Reorganization

1. Install the package in editable mode from `training/CBD` using `pip install -e .`.
2. Verify that scripts such as `torch_DG_regression_angle.py` import modules from `CBDtorch` without path tweaks.
3. Rerun any previous experiment; the logs should mirror earlier results despite the new directory layout.

## Summary of the One Hundred Sixty-Seventh Commit

New regression trials targeted the first dataset without demographic inputs. The qsub scripts were retuned for a 35-hour wall time and log under directories like `try_20220610_dense_woDG_첫번째데이터_1layer`. Both regression Python files were heavily edited: experiment names now include the date and dataset identifier, and the training loops feature expanded print statements for grid-search counters. The dense regressor definition was also reformatted for clarity.

### Why Run a Non-Demographic Baseline?

Comparing models trained with and without demographic fields reveals whether those extra inputs truly enhance prediction accuracy. By reverting to the first dataset and stripping demographics, the team established a control run against which the new demographic pipeline can be measured.

### Reproducing the Baseline Test

1. Ensure the `IWALQQ_AE_1st` dataset is present and that demographic columns are ignored when loading.
2. Submit the modified qsub scripts or execute the Python files locally. Logs should appear in the `try_20220610_dense_woDG_첫번째데이터_1layer` folders.
3. After training, compare nRMSE values against the demographic-enabled runs to assess the impact of the additional features.

## Summary of the One Hundred Sixty-Eighth Commit

A minor typo in the data-quality script `2_Data_PDFViewNCheck.py` was corrected. The on-screen help text now lists the key `'s'` for postponing a file review instead of the previous mistaken `'c'`.

### Why Fix the Help Prompt?

The PDF-check utility guides curators through large datasets, marking files for inclusion or exclusion. Accurate instructions prevent confusion when operators postpone reviewing a particular trial.

### Reproducing the Fix

Run the script in `CHK` mode and observe that the console now prints `'s - for postpone'` in the command summary. This matches the actual keyboard binding used to defer processing a record.

## Summary of the One Hundred Sixty-Ninth Commit

The developers discovered that the earlier typo fix in `2_Data_PDFViewNCheck.py` had not propagated to every branch. This commit repeats the change so the console instructions consistently show `'s' - for postpone` when reviewing PDFs. Only one print statement changed, but it prevents confusion for operators skimming hundreds of files.

### Why Correct the Script Again?

A merge from another contributor reintroduced the old message displaying `'c'` for postponement. Restoring `'s'` ensures that curators press the correct key when they need to revisit a trial later.

### Reproducing the Typo Correction

Run `python preperation/2_Data_PDFViewNCheck.py --mode CHK`. The help text printed at startup should list `'s'` as the postpone key.

## Summary of the One Hundred Seventieth Commit

This commit simply synchronized the repository with the upstream `main` branch. No files changed beyond the merge metadata, but recording the merge keeps the history intact for subsequent pulls.

## Summary of the One Hundred Seventy-First Commit

After a regression job stopped prematurely, the team extended several qsub scripts so the remaining epochs could finish. Wall times were increased from 35 to 48 hours and new echo statements record the job name, ID, and start date in the logs. Output folders gained an `_add` suffix to distinguish these continuations from the original runs. The list of embedding dimensions was trimmed to `[40, 50, 60, 70, 80]` for the non‑demographic regressors and `[50, 60, 70, 80]` for the demographic versions.

### Why Resume the Interrupted Training?

Hardware outages had halted earlier jobs before convergence. By relaunching with longer time limits and clearer logging, the researchers ensured that every combination of hyperparameters completed at least once.

### Reproducing the Extended Runs

1. From `training/MODEL/Pytorch_AE_LSTM`, submit the updated qsub scripts such as `qsub_torch_regression_angle.sh`.
2. Monitor the log directories named `*_add` for progress messages including the printed job information.
3. Verify that the embedding-dimension loops skip the smaller values and that training proceeds for the full 48-hour allocation.

## Summary of the One Hundred Seventy-Second Commit

A tiny follow‑up adjusted the demographic regression script so its embedding search begins at 50 dimensions. The earlier attempt with smaller embeddings failed to converge, so this commit focuses the compute budget on the more promising sizes.

### Reproducing the Adjusted Search

Run `training/MODEL/Pytorch_AE_LSTMwithDemographic/torch_DG_regression_angle.py` and confirm that only four embedding dimensions—50 through 80—are tested.

## Summary of the One Hundred Seventy-Third Commit

Finally, another merge from the upstream repository incorporated remote edits while preserving the help‑text fix in `2_Data_PDFViewNCheck.py`. This ensures all collaborators share the corrected script as they continue their experiments.

## Summary of the One Hundred Seventy-Fourth Commit

The researchers began another series of regression trials using the second dataset variant and introducing explicit L2 regularization.  Both `torch_DG_regression_angle.py` and `torch_DG_regression_moBWHT.py` now accept a `weight_decay` parameter passed to the `NAdam` optimizer.  The job scripts `qsub_torch_DG_regression_angle.sh` and `qsub_torch_DG_regression_moBWHT.sh` were also updated so results write to `20220619_weightDecay` directories.  This run explored a wider range of embedding dimensions—from 5 all the way to 80—to see whether weight decay of `0.01` helped or hurt model capacity.

### Why Experiment with Weight Decay?

Earlier jobs suggested the regressors might be overfitting the limited training data.  By testing L2 regularization with the new dataset split, the team hoped to stabilize training loss and improve generalization to held‑out folds.

### Reproducing the Weight-Decay Study

1. Navigate to `training/MODEL/Pytorch_AE_LSTMwithDemographic`.
2. Submit either `qsub_torch_DG_regression_angle.sh` or `qsub_torch_DG_regression_moBWHT.sh` depending on the target variable.
3. Check the `result_qsub/dgregAng/20220619_weightDecay` or `result_qsub/dgregmo/20220619_weightDecay` folders for TensorBoard logs and verify the weight‑decay value is `0.01` in the printed hyperparameters.

## Summary of the One Hundred Seventy-Fifth Commit

While reviewing hundreds of PDF plots, the team discovered that the file-viewing utility occasionally crashed when exporting selected trials.  The entire script `2_Data_PDFViewNCheck.py` was reformatted for clarity, with argument parsing split across multiple lines and all file paths converted to raw strings.  Edge cases in the export loop were hardened so that included, excluded, and postponed files copy to the correct directories without raising errors.

### Why Fix the Viewer Script?

Accurate manual inspection of raw sensor traces is crucial for building trustworthy datasets.  Any hiccup in the viewer slows down curation, so this commit eliminates the glitches encountered when saving or skipping files.

### Reproducing the Viewer Workflow

Run `python preperation/2_Data_PDFViewNCheck.py --mode CHK` and step through a few trials.  Confirm that pressing the designated keys moves PDFs to the appropriate folders and that no exceptions appear in the console.

## Summary of the One Hundred Seventy-Sixth Commit

Initial runs with weight decay did not converge as expected, so the learning rate was temporarily doubled from `0.001` to `0.002`.  The qsub scripts now log results under `20220620_weightDecay` and request a shorter 24‑hour wall time after monitoring resource usage.  Both regression Python files were updated so their learning‑rate lists start at `0.002`.

### Why Increase the Learning Rate?

Training curves plateaued early in the previous attempt, suggesting the optimizer needed a larger step size.  This change tested whether doubling the rate could kick the models out of local minima.

### Reproducing the Higher‑LR Run

Submit the same qsub scripts as before and verify that new subfolders named `20220620_weightDecay` appear under `result_qsub`.  TensorBoard should show the learning rate of `0.002` in each experiment directory.



## Summary of the One Hundred Seventy-Seventh Commit

To keep the repository in sync with the public `main` branch, the developers performed a merge that brought in upstream changes from other collaborators. The merge preserved the recent viewer-script fixes while incorporating unrelated updates, ensuring everyone continued from a common baseline.

### Why Perform the Merge?

By merging regularly, the team avoided accumulating conflicts and guaranteed that future patches applied cleanly. This commit shows the project was actively maintained in coordination with the upstream repository.

### Verifying the Merge

Run `git log --merges` and locate commit `dba0988` to confirm this merge occurred. Check that `2_Data_PDFViewNCheck.py` still contains the formatting fixes from the previous commit.

## Summary of the One Hundred Seventy-Eighth Commit

Results from the high learning-rate experiment were disappointing. Training became unstable, so the learning rate was restored to `0.001` and the L2 `weight_decay` was lowered from `0.01` to `0.005`. Job directories now embed the decay value so different runs are easy to distinguish.

### Why Adjust the Regularization?

Excessive weight decay can over-penalize the model while too large a learning rate causes oscillation. This commit strikes a balance by returning to the original rate but testing a smaller decay to encourage smoother convergence.

### Reproducing the Revised Decay Setting

1. Edit the regression scripts so `list_learningRate = [0.001]` and `weight_decay = 0.005`.
2. Submit the qsub jobs and verify that logs write to folders containing `weightDecay_0.005`.
3. Compare TensorBoard curves with those from the previous decay to judge improvement.

## Summary of the One Hundred Seventy-Ninth Commit

Training had stabilized, so focus shifted to evaluating the regressor on every trial. The new notebook `makeEstimation_VAEwithDG.ipynb` loads a saved checkpoint and predicts knee angles across the full `IWALQQ_AE_1st` dataset. The notebook saves one spreadsheet per trial, arranged by fold and embedding dimension. `dense_dg.py` was updated so the model can be reloaded on systems without a GPU.

### Why Export Predictions?

Storing each trial's output allows manual inspection and future analyses without rerunning the model. It also provides a reference in case training code changes later.

### Reproducing the Inference Export

1. Open the notebook and set `model_path` to your trained model.
2. Execute all cells to produce `.xlsx` files in directories labeled by fold and latent size.
3. Spot-check a few spreadsheets to confirm each contains a full angle sequence.

## Summary of the One Hundred Eightieth Commit

The estimation notebook was further expanded to compute step-wise errors between the predicted and true angles. It reads all spreadsheets from the previous commit, aligns them with ground-truth data, and writes summary files like `result_0_fold.xlsx` that list predictions, targets, and RMSE values.

### Why Compare Predictions with Ground Truth?

Quantitative error metrics reveal whether the model generalizes beyond the validation folds and highlight subjects or phases where performance lags.

### Reproducing the Comparison

1. After generating the raw prediction spreadsheets, run the new analysis cells.
2. Ensure `result_*_fold.xlsx` files appear next to the individual trial outputs.
3. Open these summaries to review average errors across each fold.

## Summary of the One Hundred Eighty-First Commit

To make experiment tracking clearer, the regression scripts now log the `weight_decay` hyperparameter to TensorBoard, and a typographical error in the learning-rate variable was corrected. These small fixes guarantee that every run records the exact settings used.

### Why Track `weight_decay`?

Without logging this value it would be impossible to tell which decay produced a given checkpoint, complicating later comparisons.

### Reproducing the Logging Fix

Resubmit any regression jobs and open TensorBoard’s HParams tab to confirm that `weight_decay` is listed with the intended value.

## Summary of the One Hundred Eighty-Second Commit

The researchers briefly questioned whether the K-fold splits covered every subject. The notebook `StudyRoom/whoinTestSet.ipynb` enumerates which participants appear in each test set, while `tmp2_speed_check.ipynb` was adjusted so intermediate CSV files load properly. Although the check proved redundant, documenting it prevents future confusion.

### Why Double-Check Fold Composition?

Ensuring that no subject was accidentally omitted gives confidence that evaluation metrics truly reflect the entire dataset.

### Reproducing the Fold Analysis

Run `whoinTestSet.ipynb` and compare the printed lists of IDs with your own dataset to confirm the splits.

## Summary of the One Hundred Eighty-Third Commit

Finally, the same prediction pipeline was applied to the knee-moment target `moBWHT`. The notebook now accepts a moment model and saves hundreds of spreadsheets under `moBWHT/70`, where `70` denotes the embedding dimension. This provides a complete record of moment predictions for every trial.

### Why Predict Moments?

Evaluating both joint angles and moments demonstrates the versatility of the VAE-LSTM regressor and supports downstream biomechanics studies.

### Reproducing the Moment Predictions

1. Set `model_path` in the notebook to a checkpoint trained on `moBWHT`.
2. Execute all cells to generate the `.xlsx` files under `moBWHT/70`.
3. Compare these moment predictions with the angle outputs to assess overall model performance.

## Summary of the One Hundred Eighty-Fourth Commit

The previous run used a weight-decay value of 0.005, which caused the VAE-LSTM regressors to converge slowly. This commit reduces the L2 regularization strength to **0.001** in both the angle and moment training scripts. The qsub wrappers were also updated so their output folders include `weightDecay_0.001`, keeping experiment logs organized by hyperparameter.

### Why Lower the Decay?

Early TensorBoard curves indicated that 0.005 penalized the model too heavily, hindering learning. Dropping to 0.001 lets the optimizer explore smoother solutions while still discouraging overfitting.

### Reproducing the 0.001 Run

1. Edit the regression Python files so `weight_decay = 0.001`.
2. Submit `qsub_torch_DG_regression_angle.sh` and `qsub_torch_DG_regression_moBWHT.sh`.
3. Check that results write to `20220620_weightDecay_0.001` and monitor the loss curves in TensorBoard.

## Summary of the One Hundred Eighty-Fifth Commit

After inspecting the 0.001 results, the team experimented with an even smaller `weight_decay` of **0.0005**. The same four files were modified, and job directories now end with `weightDecay_0.0005`.

### Why Try 0.0005?

The 0.001 setting reduced overfitting somewhat, but validation error still climbed late in training. Testing 0.0005 helps determine whether a lighter penalty improves generalization without sacrificing stability.

### Reproducing the 0.0005 Run

1. Set `weight_decay = 0.0005` in both regression scripts.
2. Resubmit the qsub jobs to generate logs under `20220620_weightDecay_0.0005`.
3. Compare these curves against the previous run to judge the effect.

## Summary of the One Hundred Eighty-Sixth Commit

The search continued by lowering the decay once more to **0.00025**. All regression scripts and qsub wrappers were updated accordingly so that outputs fall under `weightDecay_0.00025`.

### Why Go Even Lower?

The researchers hypothesized that tiny amounts of weight decay might strike the best tradeoff between bias and variance. Trying 0.00025 lets them observe whether the trend of decreasing decay keeps improving validation loss.

### Reproducing the 0.00025 Run

1. Change `weight_decay` to `0.00025` in the two training scripts.
2. Submit the qsub files and verify new directories labelled `weightDecay_0.00025`.
3. Examine TensorBoard to see if the reduced regularization yields smoother learning curves.

## Summary of the One Hundred Eighty-Seventh Commit

With training complete, attention turned to analyzing absolute errors. The notebook `pdfGen.ipynb` was introduced to aggregate per-trial spreadsheets, compute mean and standard deviation across folds, and plot the results into a single PDF named `Total_result_moBWHT.pdf`. Numerous `.xlsx` files in `estimation/IWALQQ_AE_1st/` were regenerated, reflecting the latest models.

### Why Produce Error Figures?

Visualizing the distribution of errors helps identify systematic biases and communicates performance to non-programmers. The PDF consolidates hundreds of trials into concise charts that highlight typical accuracy.

### Reproducing the Figures

1. Run `makeEstimation_VAEwithDG.ipynb` to ensure the latest prediction spreadsheets are present.
2. Execute every cell in `pdfGen.ipynb` to create `Total_result_moBWHT.pdf` under the dataset folder.
3. Review the PDF to inspect error trends across embedding dimensions and folds.

## Summary of the One Hundred Eighty-Eighth Commit

Finally, a new notebook `maketable.ipynb` was added to summarize all results in tabular form. It scans exported CSVs, computes mean and standard deviation for each metric, and writes the consolidated table to `result_4_condition.xlsx`.

### Why Summarize in Tables?

While graphs offer visual intuition, numeric tables make it easy to compare models quantitatively or feed the data into other analyses. Automating this step ensures consistency across experiments.

### Reproducing the Table

1. Place all result CSV files from previous runs in the same directory as `maketable.ipynb`.
2. Open the notebook and run each cell; it will create `result_4_condition.xlsx`.
3. Use this spreadsheet to rank models by nRMSE and decide which configuration to pursue next.

## Summary of the One Hundred Eighty-Ninth Commit

The hyperparameter search for an ideal regularization strength continued. After testing 0.00025, the developers reduced `weight_decay` further to **0.0001** in both demographic regressors. Each qsub script was updated so logs would save under `20220620_weightDecay_0.0001` and the Python training files were modified accordingly. The goal was to see if almost-negligible L2 penalties could still curb overfitting without impeding convergence.

### Why Test 0.0001?

Previous runs showed that 0.00025 produced the lowest validation loss so far, but the curves were still noisy near the end of training. Pushing to 0.0001 let the team measure whether the trend continued or if the model began to overfit. This tiny decay essentially served as a control to verify the sensitivity of the network to regularization.

### Reproducing the 0.0001 Experiment

1. Open `torch_DG_regression_angle.py` and `torch_DG_regression_moBWHT.py` and set `weight_decay = 0.0001`.
2. Adjust the corresponding output paths in both `qsub_torch_DG_regression_*` scripts so the folder names end with `weightDecay_0.0001`.
3. Submit the jobs via `qsub` and monitor `result_qsub/dgregAng/20220620_weightDecay_0.0001` and `result_qsub/dgregmo/20220620_weightDecay_0.0001`.
4. Compare the TensorBoard logs against earlier runs to judge whether this minimal decay improves or harms generalization.

## Summary of the One Hundred Ninetieth Commit

Work then shifted to generating new training datasets. The preprocessing notebooks were revised to create **IWALQQ_AE_3rd**, which mirrors the first dataset but uses the axis‑corrected raw files located in `RAW_AXIS_corrected`. Various notebook cells were rerun under the updated Python 3.8.13 environment, and metadata fields such as execution counts and interpreter hashes changed accordingly.

### Why Another Dataset?

Switching to the corrected raw files removed residual orientation errors that had crept into earlier versions. By keeping the random seed at 41 but using the improved source data, the team produced a cleaner dataset without altering the overall train/test splits. This allowed fair comparisons to the `IWALQQ_AE_1st` baseline while benefiting from the axis fixes.

### Reproducing IWALQQ_AE_3rd

1. In `4_DataSet_IWALQQ_AE.ipynb`, set `seed_rand = 41` and `nameDataset = 'IWALQQ_AE_3rd'`.
2. Execute every cell starting from the raw data filtering steps to regenerate the `.npz` files under `preperation/SAVE_dataSet/IWALQQ_AE_3rd`.
3. Ensure that `1_Data_Checker.ipynb`, `2_Data_PDFViewNCheck.py`, `3_0_Data_filtertoSave.ipynb`, and `3_1_Data_timenormalized.ipynb` all point to the `RAW_AXIS_corrected` directories so no misaligned sensors slip through.

## Summary of the One Hundred Ninety-First Commit

A companion dataset named **IWALQQ_AE_4th** followed. This version copies the fold composition of `IWALQQ_AE_2nd` but is regenerated from the same corrected raw files. The notebook `4_DataSet_IWALQQ_AE.ipynb` now sets `seed_rand = 777` and saves under the new name. The `list_dataset_correction.xlsx` tracker was updated as well.

### Why Mirror the Second Dataset?

Using the fourth dataset let researchers gauge whether improvements from the axis correction held across different subject splits. By matching the second dataset’s structure, they could compare models trained on similar demographics while benefiting from cleaner preprocessing.

### Reproducing IWALQQ_AE_4th

1. Set `seed_rand = 777` and `nameDataset = 'IWALQQ_AE_4th'` in the same notebook.
2. Rerun all preprocessing cells to produce the dataset under `preperation/SAVE_dataSet/IWALQQ_AE_4th`.
3. Update `list_dataset_correction.xlsx` to record the creation date and any notes about the axis corrections.

## Summary of the One Hundred Ninety-Second Commit

With both new datasets available, the VAE‑LSTM training script was retuned to consume `IWALQQ_AE_4th`. The qsub wrapper `qsub_torch_vaelstm.sh` now submits a job named `VL_4th` and writes logs to `20220628_IWALQQ_AE_4th`. Inside `torch_VAE_LSTM.py`, the `exp_name` and `nameDataset` variables were changed accordingly so that checkpoints and TensorBoard events match the dataset identifier.

### Why Retrain the VAE‑LSTM?

Earlier models were trained on the second dataset and could not exploit the freshly corrected IMU signals. By rerunning the autoencoder on the fourth dataset, the team hoped to generate better latent features for downstream regression tasks.

### Reproducing the VAE‑LSTM Run

1. Edit `torch_VAE_LSTM.py` so `nameDataset = 'IWALQQ_AE_4th'` and update `exp_name` to reflect the date.
2. Modify `qsub_torch_vaelstm.sh` with the same experiment name and output directory.
3. Submit the job and monitor `result_qsub/vaelstm/20220628_IWALQQ_AE_4th` for training progress.
4. Once finished, verify that the saved `.pt` files correspond to the new dataset.

## Summary of the One Hundred Ninety-Third Commit

The final commit in this series re‑ran the demographic regressors using the newly trained VAE on the fourth dataset. Both `torch_DG_regression_angle.py` and `torch_DG_regression_moBWHT.py` were updated to load `IWALQQ_AE_4th`, set `weight_decay = 0.001`, and reduce training to **1500 epochs**. Their qsub scripts now point to result folders dated `20220628_weightDecay_0.001`.

### Why Revisit the Regressors?

Switching datasets invalidated earlier conclusions about regularization. By coupling the revised VAE embeddings with a mid‑range decay of 0.001, the researchers aimed to establish a new baseline on the cleaned data. Shortening the epoch count kept runtimes manageable during this exploratory phase.

### Reproducing the Updated Regressors

1. Ensure the VAE checkpoints from the previous step are available under the expected model version.
2. Set `nameDataset = 'IWALQQ_AE_4th'` and `weight_decay = 0.001` in both regression scripts.
3. Submit the updated qsub files and watch the logs under `dgregAng/20220628_weightDecay_0.001` and `dgregmo/20220628_weightDecay_0.001`.
4. Inspect TensorBoard to compare these results with the earlier 0.0001 and 0.00025 runs.
## Summary of the One Hundred Ninety-Fourth Commit

The codebase introduced a "mini" demographic regressor that attaches a trimmed
set of subject features to the VAE embeddings. A new module
`training/CBD/CBDtorch/dense_dg_mini.py` defines this lighter network, while
`torch_DGMini_regression_angle.py` and `torch_DGMini_regression_moBWHT.py`
train it on knee angles and moments respectively. Matching qsub wrappers were
added so jobs can run on the BU SCC cluster. Several study notebooks were also
committed to document the architecture and compare results, and summary tables
appear under `estimation/Compare_IWALQQ_AE_4th`.

### Why Build the Mini Demographic Regressor?

Earlier experiments used a full demographic vector that included every
available field. The team wanted to know whether fewer fields could achieve
similar accuracy with less overfitting. The new mini model keeps only the most
influential demographics, reducing dimensionality and training time.

### Reproducing the Mini-Regressor Experiment

1. Install the project dependencies and ensure the `IWALQQ_AE_4th` dataset is
   available.
2. Launch `qsub_torch_DGMini_regression_angle.sh` or
   `qsub_torch_DGMini_regression_moBWHT.sh` to queue the jobs.
3. Monitor the result folders indicated inside the scripts for logs and saved
   checkpoints.
4. Use `maketable_withDecay.ipynb` to aggregate the new runs and compare them to
   the full demographic regressors.

## Summary of the One Hundred Ninety-Fifth Commit

At the end of the day the researchers reorganized their estimation workflow.
The old `pdfGen.ipynb` notebook was deleted and replaced by two specialized
versions: `pdfGen_whole.ipynb` generates full-report PDFs while
`pdfGen_onlyDiff.ipynb` highlights just the differences between models. Result
folders now contain summary PDFs such as `WHOLE_Total_result_angle.pdf` and
`WHOLE_Total_result_moBWHT.pdf`. The original `maketable.ipynb` was renamed to
`maketable_withDecay.ipynb` so tables indicate which weight-decay setting they
represent.

### Why Clean Up Estimation and PDFs?

Weight-decay sweeps produced many intermediate logs. By streamlining the
notebooks and adding clearly named summary PDFs, collaborators can quickly grasp
how each setting performed. Removing stray files like the old
`IWALQQ_AE_1st` spreadsheets kept the repository organized for the next phase of
analysis.

### Reproducing the Updated PDF Workflows

1. After running the regression scripts, open `pdfGen_whole.ipynb` to create a
   full comparison report or `pdfGen_onlyDiff.ipynb` for a condensed view.
2. Execute `maketable_withDecay.ipynb` to regenerate the CSV tables used in the
   PDFs.
3. Verify that the generated PDFs appear under the appropriate result
   directories alongside the new tables.

## Summary of the One Hundred Ninety-Sixth Commit

This commit finalized the training scripts after all models completed. The
`maketable.ipynb` notebook for comparing weight-decay runs was moved into the
root result folder, and minor path corrections were applied across the qsub
wrappers for both the full and mini demographic models. Code comments were
cleaned so future runs reference the standardized directory layout.

### Why Consolidate the Scripts?

With every variant trained, the team wanted a single place to examine results.
Relocating the notebook and harmonizing paths prevents confusion when rerunning
jobs or sharing logs with collaborators.

### Reproducing the Finalized Setup

1. Use the updated qsub scripts without modification—they already point to the
   finalized directories.
2. Run `maketable.ipynb` from the result folder to collate the metrics from all
   completed runs.

## Summary of the One Hundred Ninety-Seventh Commit

Large batches of prediction spreadsheets were committed for both angles and
moments using the `IWALQQ_AE_4th` dataset. Each subject’s results now reside in
fold-specific folders alongside `TruePredDiff.xlsx` summaries and overall
`WHOLE_Total_result` PDFs. The dense regressor code received a minor update so
these exports include every epoch’s best model.

### Why Capture Every Prediction?

Final evaluation required examining individual subjects and plotting aggregate
metrics. By storing per-subject spreadsheets, the researchers can audit cases
where the model performs unusually well or poorly and prepare figures for future
reports.

### Reproducing the Full Export

1. Run the estimation notebooks such as `makeEstimationWithPDF_Dense.ipynb`
   after training completes.
2. Ensure the paths inside these notebooks point to the `IWALQQ_AE_4th`
   checkpoints and dataset directories.
3. Execute all cells to write the spreadsheets and summary PDFs shown in this
   commit.

## Summary of the One Hundred Ninety-Eighth Commit

Further analysis revealed a minor axis misalignment in the moment predictions.
All four estimation notebooks were updated to correct the moBWHT orientation and
the resulting spreadsheets and PDFs were regenerated. Only the data under
`IWALQQ_AE_4th/moBWHT` changed, but the fix propagates through the comparison
tables and final reports.

### Why Fix the moBWHT Axis?

Accurate joint-moment orientation is critical for clinical interpretation. The
team noticed subtle sign errors when plotting the earlier results and traced the
issue to a coordinate mismatch. Correcting the axis ensures that peak values and
impulse calculations reflect the true biomechanical directions.

### Reproducing the Axis Correction

1. Open the updated `makeEstimationWithPDF_*.ipynb` notebooks and rerun them on
   your machine.
2. Replace any previously generated `TruePredDiff.xlsx` files and PDFs with the
   new versions.
3. Double-check a few sample plots to confirm the moments now align with the
   expected coordinate system.

## Summary of the One Hundred Ninety-Ninth Commit

The estimation pipeline was reorganized so PDF reports and spreadsheets could be generated with less manual editing.  A small helper module `dir_CBD.py` now creates result folders on demand while `plot_CBD.py` collects all of the Matplotlib plotting utilities.  Another file, `scaler_CBD.py`, centralizes the sensor-wise scaling logic.  The main notebook `makeEstimationWithPDF_Dense.ipynb` was trimmed down to import these utilities, and the older `makeEstimationwithPDF_wDgMini.ipynb` was removed entirely.  These changes drastically reduced duplicated code and ensured that every experiment used the same plotting routines.

### Why Consolidate the Estimation Code?

Earlier notebooks had each reimplemented directory setup, plotting, and scaling in slightly different ways.  This made it easy for results to land in inconsistent folders or for plots to be generated with mismatched axis limits.  By extracting common functions into standalone modules the team could produce cleaner figures and keep their results tree organized.

### Reproducing the Cleaned Estimation Process

1. Ensure the new `estimation` modules are on your Python path by running from the repository root.
2. Open `makeEstimationWithPDF_Dense.ipynb` and execute all cells.  The notebook will create any missing directories and save PDFs using the shared plotting functions.
3. Check the resulting folders under `estimation/` to verify that spreadsheets and PDF reports appear without manual intervention.

## Summary of the Two Hundredth Commit

A short merge commit resolved conflicts after pulling the previous axis-fix changes.  Several generated spreadsheets and PDFs were updated to reflect the corrected moBWHT orientation.  The estimation notebooks received minor path tweaks so they load the shared helper modules introduced earlier.

### Why Merge Immediately After the Cleanup?

The axis correction touched many of the same result files that the new estimation modules used.  Pulling and merging ensured that the repository held only one canonical set of PDFs and that all notebooks referenced the fixed data.

### Reproducing the Merged Results

Simply rerun `makeEstimationWithPDF_Dense.ipynb` after pulling this revision.  You should obtain PDFs identical to those stored in the repository, confirming the merge applied cleanly.

## Summary of the Two Hundred First Commit

A brand new preprocessing notebook, `4_DataSet_IWALQQ_AE_MOSTyle.ipynb`, was added.  It converts the existing IMU files into a format styled after the public MOST dataset so future models can be compared against external benchmarks.  The notebook walks through loading each subject, applying the usual axis corrections, and saving the processed arrays under `preperation/SAVE_dataSet/IWALQQ_AE_MOSTyle`.

### Why Create a MOSTyle Dataset?

Previous datasets used project-specific naming conventions that differed from the MOST format.  By mirroring MOST’s structure, the team can train models that are directly comparable to studies in the literature and potentially reuse code written for MOST without modification.

### Reproducing the MOSTyle Dataset

1. Open `preperation/4_DataSet_IWALQQ_AE_MOSTyle.ipynb`.
2. Execute all cells to generate the new `.npz` files under `preperation/SAVE_dataSet/IWALQQ_AE_MOSTyle`.
3. Record the creation in `list_dataset_correction.xlsx` if you are tracking dataset versions.

## Summary of the Two Hundred Second Commit

Training scripts for the new MOSTyle dataset were introduced.  Fresh qsub wrappers—`qsub_MOSTyle_torch_vaelstm.sh` and two regression counterparts—submit VAE-LSTM and demographic regressor jobs on the BU SCC cluster.  Corresponding Python files (`torch_MOSTyle_VAE_LSTM.py`, `torch_MOSTyle_DG_regression_angle.py`, and `torch_MOSTyle_DG_regression_moBWHT.py`) closely follow the existing fourth-dataset scripts but load data from `IWALQQ_AE_MOSTyle`.

### Why Start MOSTyle Training?

With the dataset prepared, the researchers needed a full set of experiments to gauge whether the MOST-style formatting changed model behavior.  Separate job scripts keep these runs distinct from the earlier datasets, making it easy to compare learning curves and final metrics.

### Reproducing the MOSTyle Runs

1. Submit `qsub_MOSTyle_torch_vaelstm.sh` to train the autoencoder.
2. After it completes, launch the two demographic regressor scripts with `qsub` as well.
3. Monitor the new result directories specified inside the scripts for TensorBoard logs and checkpoints.

## Summary of the Two Hundred Third Commit

A minor error in `qsub_torch_vaelstm.sh` prevented the MOSTyle autoencoder from launching properly.  This commit corrected the job name and output paths, and switched the executed script back to `torch_VAE_LSTM.py`.  Only a few lines changed, but without them the queue submission failed immediately.

### Why Fix the Qsub Script?

Testing revealed that the previous job configuration still referenced the old experiment directories.  Adjusting the script avoided clobbering earlier results and ensured logs accumulated in the correct folder.

### Reproducing the Fixed Submission

1. Use the updated `training/MODEL/Pytorch_AE_LSTM/qsub_torch_vaelstm.sh` without further edits.
2. Submit it via `qsub` and confirm that the job appears as `VL_4th` with output under `result_qsub/vaelstm/20220628_IWALQQ_AE_4th`.


## Summary of the Two Hundred Fourth Commit

The scripts for demographic regressors using the MOSTyle dataset were updated so their experiment names, dataset references, and search grids match the final autoencoder outputs.  The angle and moment regressors now load `IWALQQ_AE_MOSTyle_2nd` and search over embedding dimensions from five to eighty.  This commit reflects the team's first pass at examining MOSTyle results in depth, making sure the hyperparameters align with prior runs.

### Why Revisit the Regression Scripts?

Early MOSTyle experiments reused settings from the fourth dataset.  Renaming the models and expanding the embedding dimension sweep prevented confusion when comparing logs.  The smaller input size (three sensors per sequence) also required updating `num_features` so the dense layers receive the correct shape.

### Reproducing the MOSTyle Regression Checks

1. Edit nothing—use the revised Python files under `training/MODEL/Pytorch_AE_MOSTylewithDemographic`.
2. Submit the associated qsub scripts to start new angle and moment regressions.
3. Monitor the output folders defined inside each script for checkpoints and TensorBoard logs.

## Summary of the Two Hundred Fifth Commit

Two qsub scripts were adjusted for a new experiment that trains the VAE‑LSTM and demographic regressors using only three sensors.  Wall times were trimmed to twelve hours and the job names changed to `MORADG` and `MORMDG` respectively.  These updates mark the beginning of a lightweight model variant aimed at faster turnaround.

### Why Limit to Three Sensors?

Reducing the sensor count tests whether comparable accuracy can be achieved with minimal hardware, which is valuable for portable applications.  Shorter runtimes also speed up grid searches when exploring new regularization settings.

### Reproducing the Three‑Sensor Runs

1. Submit `qsub_MOSTyle_torch_DG_regression_angle.sh` and `qsub_MOSTyle_torch_DG_regression_moBWHT.sh`.
2. Confirm that each job requests twelve hours and writes logs under the `MOSTyle_dgreg` directories.
3. Check the resulting checkpoints to see how three‑sensor inputs affect performance.

## Summary of the Two Hundred Sixth Commit

A wide-ranging update produced hundreds of per-subject Excel workbooks summarizing the MOSTyle regressors.  New estimation notebooks, such as `makeEstimationwithPDF_wDgMOSTyle.ipynb`, generate PDF reports from these spreadsheets.  Additional scripts and qsub wrappers under `EXPERIMENTAL_Pytorch_AE_LSTMwithDemographic` automate further grid searches.  The commit also recorded large prediction tables (`wDgMOSTyle_hparams_table.csv`) for later analysis.

### Why Export So Many Spreadsheets?

The researchers needed a granular look at prediction accuracy across all subjects and folds.  Saving each case to its own file enabled rapid plotting and manual inspection without rerunning the model.  The new notebooks then collate these results into PDFs for sharing with collaborators.

### Reproducing the Export and PDF Generation

1. After training completes, run `makeEstimationwithPDF_wDgMOSTyle.ipynb`.
2. Inspect the generated `TruePredDiff.xlsx` files and compiled PDF summaries under the corresponding result folders.
3. Use the EXP scripts and qsub wrappers if you wish to replicate the extended grid search with these exports enabled.

## Summary of the Two Hundred Seventh Commit

The dense demographic regressor architecture was overhauled to include larger fully connected layers and heavier dropout.  The experimental script and its qsub submission file were updated accordingly.  These changes aimed to combat overfitting observed in earlier MOSTyle runs.

### Why Increase Model Capacity?

Initial tests suggested the regressor underfit when trained on the reduced sensor set.  By expanding the hidden layers from 1,536 to 4,096 units and raising dropout to 50%, the team sought a better balance between capacity and regularization.

### Reproducing the Revised Regressor

1. Use the latest `dense_dg_experimental.py` and `EXP_torch_DG_regression_angle.py` from `training/CBD` and `training/MODEL/EXPERIMENTAL_Pytorch_AE_LSTMwithDemographic`.
2. Submit the `qsub_EXP_torch_DG_regression_angle.sh` script.
3. Compare TensorBoard curves against prior runs to evaluate the deeper network.

## Summary of the Two Hundred Eighth Commit

Several estimation notebooks were cleaned of obsolete plotting code and intermediate cells.  The refactored versions load data faster and focus solely on generating the final error PDFs.  No functional changes were made to the underlying calculations—this commit strictly removes clutter.

### Why Tidy the Estimation Notebooks?

Over months of experimentation the notebooks accumulated debugging cells that slowed execution and obscured the main workflow.  Streamlining them keeps the repository manageable and ensures new users can reproduce the reports without wading through outdated steps.

### Reproducing the Clean Notebook Outputs

1. Open any of the updated notebooks under `estimation/`.
2. Run all cells to confirm that the plots and PDFs match the earlier versions.
3. Because the heavy intermediate code was deleted, these notebooks should execute noticeably faster.


## Summary of the Two Hundred Ninth Commit

The repository expanded its post-processing capabilities with a dedicated peak-detection pipeline. New modules under `estimation/module` parse prediction spreadsheets and compute moment peaks across the gait cycle. The script `estimation/peak_detection.py` orchestrates the workflow, reading CSV outputs from previous regression runs and saving the detected maxima to fresh Excel workbooks under `estimation/Result_peak/`. Four example files capture results for dense and demographic models with or without mini datasets.

### Why Add Peak Detection?

Peak knee moments are important clinical metrics for assessing joint loading. Earlier scripts only provided full time-series predictions, leaving collaborators to manually inspect curves. Automating peak extraction simplifies comparisons across subjects and models, enabling quick statistical analysis of maximum joint stress.

### Reproducing the Peak Analysis

1. Ensure you have prediction CSVs generated by the regression scripts.
2. Run `python estimation/peak_detection.py` to parse these files and compute per-subject peak values.
3. Verify that spreadsheets like `peak_DenseModel.xlsx` appear under `estimation/Result_peak/` with columns listing each fold's detected peaks.

## Summary of the Two Hundred Tenth Commit

Building on the peak workflow, the project introduced impulse calculations for knee moments. A new module `estimation/module/impulse.py` integrates with `moment.py` to integrate the absolute moment curve over time. The `impulse_calculation.py` script processes prediction spreadsheets and outputs summaries in the `estimation/Result_impulse/` directory. Existing peak files were lightly updated to match the new parsing logic.

### Why Measure Impulse?

While peak values capture instantaneous loads, the impulse quantifies cumulative joint stress during the stance phase. This additional metric helps evaluate how different models track overall moment patterns and whether certain conditions lead to higher loading over the gait cycle.

### Reproducing the Impulse Calculation

1. After running the regressors, execute `python estimation/impulse_calculation.py`.
2. Check the newly created Excel files in `estimation/Result_impulse/` for per-fold impulse totals.
3. Optionally rerun `peak_detection.py` to keep the peak and impulse summaries synchronized.

## Summary of the Two Hundred Eleventh Commit

Work began on a “NOTSENSOR” variant where one or more IMU channels are intentionally omitted. A comprehensive notebook `preperation/4_DataSet_IWALQQ_AE_NOTSENSOR.ipynb` generates the reduced dataset, while `metric_NOTSENSOR.py` defines error functions tailored to the missing inputs. Numerous StudyRoom notebooks were duplicated under `training/MODEL/_NOTSENSOR/` to experiment with autoencoders, VAE‑LSTM models, and dense regressors on this altered data. Associated qsub scripts and Torch programs were added so the entire training pipeline could be executed on the BU SCC. Example GIFs and MNIST binaries were included for quick testing of the notebook code.

### Why Explore a NOTSENSOR Dataset?

The team hoped to identify the minimal sensor configuration that still captures gait dynamics. By removing channels and retraining the models, they could quantify performance loss and determine whether certain sensors contribute little to the final predictions. The extensive set of notebooks lays the groundwork for these comparisons.

### Reproducing the NOTSENSOR Setup

1. Open `preperation/4_DataSet_IWALQQ_AE_NOTSENSOR.ipynb` and execute all cells to create the reduced dataset and associated scalers.
2. Install the updated `CBDtorch` package from `training/CBD` so the new metrics are available.
3. Launch the qsub scripts under `training/MODEL/_NOTSENSOR/Pytorch_AE_LSTM*` to train autoencoders and regressors on the NOTSENSOR data.

## Summary of the Two Hundred Twelfth Commit

After assembling the NOTSENSOR scripts, the authors reviewed each training file for consistent scaling behavior. Several qsub wrappers and Python programs received small corrections so that the selected scalers were loaded properly regardless of sensor count. Affected files include the various `torch_*` regressors and their submission scripts. Some job filenames were also standardized (e.g., renaming grid search scripts to `qsub_torch_angleModel.sh`).

### Why Double-Check Scaling?

Swapping sensors changes the feature ranges fed into the network. Misaligned scalers can inflate error metrics or prevent convergence entirely. This audit ensured that every script points to the matching scaler objects generated during preparation.

### Reproducing the Scaling Fixes

1. Use the corrected qsub scripts from this commit when launching NOTSENSOR or standard runs.
2. Confirm in the console output that each job loads the expected scaler file path.
3. Compare final errors against earlier logs to verify that the scaling behaves consistently.

## Summary of the Two Hundred Thirteenth Commit

To prepare for sensor-ablation experiments, a massive batch of per-subject spreadsheets was committed under `estimation/sensorwise/`. These files store angle and moment predictions for every fold and subject in the fourth autoencoder dataset. New estimation notebooks in the `estimation/notsensor/` directory generate PDF summaries and aggregate tables from these spreadsheets. A small `.vscode` setting tweak rounds out the commit.

### Why Archive Sensorwise Results?

Analyzing each sensor configuration independently required a baseline of full-sensor predictions. By saving thousands of individual Excel files, the team could quickly slice the data by sensor group and compare against the upcoming NOTSENSOR models without rerunning heavy computations.

### Reproducing the Sensorwise Baseline

1. Ensure the standard regressors have produced their prediction spreadsheets.
2. Open the notebooks under `estimation/notsensor/` and run them to compile tables and PDFs from the sensorwise directories.
3. Use these outputs as reference when evaluating models trained with certain sensors removed.


## Summary of the Two Hundred Fourteenth Commit

This commit introduces automated aggregation scripts to consolidate the thousands of prediction spreadsheets generated in the previous sensorwise baseline. A new Python module `estimation/sensorwise/aggregate.py` walks through every subject folder, stacks the angle and moment predictions across all folds, and saves combined CSV files under `estimation/sensorwise/summary/`. Additional helper functions compute mean errors per sensor group and produce charts highlighting variations between configurations.

### Why Aggregate the Sensorwise Outputs?

Working with individual Excel files became cumbersome as the dataset grew. By merging them into a handful of summary tables, the team could quickly visualize trends and import the data into statistical software. These aggregates also serve as reference inputs for later ablation studies, ensuring that comparisons are drawn from identical prediction sets.

### Reproducing the Aggregation Process

1. Install the `pandas` and `openpyxl` packages if they are not already in your environment.
2. Run `python estimation/sensorwise/aggregate.py` from the repository root.
3. Inspect the new CSV files under `estimation/sensorwise/summary/` and verify that each sensor combination has a corresponding aggregate of all subjects.

## Summary of the Two Hundred Fifteenth Commit

Building on the sensorwise framework, the fifteenth commit formalizes a three-sensor dataset dubbed `IWALQQ_AE_3SENS`. Preparation notebooks in `preperation/` generate this reduced set by selecting the shank, thigh, and foot IMUs while dropping all others. Matching scalers are stored alongside the data, and a brief report documents how signal quality compares to the full-sensor version.

### Why Create a Three-Sensor Dataset?

Early NOTSENSOR experiments indicated that a subset of sensors might suffice for certain regression tasks. By explicitly packaging a three-sensor dataset, the researchers could benchmark performance without manual preprocessing steps. This dataset also simplifies deployment on wearable devices where fewer sensors reduce cost and setup time.

### Reproducing the Three-Sensor Data Split

1. Open `preperation/5_DataSet_IWALQQ_AE_3SENS.ipynb` and run all cells to slice the original recordings.
2. Verify that new `.npz` archives appear under `preperation/SAVE_dataSet/AE_3SENS/` with accompanying scaler pickle files.
3. Update your training scripts to point to these paths or use the provided qsub wrappers in `training/MODEL/_NOTSENSOR/3SENS/`.

## Summary of the Two Hundred Sixteenth Commit

New training scripts were added to evaluate autoencoder, VAE-LSTM, and dense regressor models on the `IWALQQ_AE_3SENS` dataset. Each script mirrors its full-sensor counterpart but imports the reduced dataset and logs results in directories named `*_3SENS`. Job submission files request slightly shorter runtimes because the smaller inputs train faster. Result tables in `estimation/Result_3SENS/` capture per-fold metrics for direct comparison with prior runs.

### Why Train Separate Models for Three Sensors?

Model behavior can change dramatically when the input dimensionality drops. Separate scripts avoid hard-coded paths and hyperparameters meant for the original data, reducing chances of accidental mix-ups. Logging to dedicated folders keeps the experiment history organized and makes it easier to plot performance deltas between sensor counts.

### Reproducing the Three-Sensor Training Runs

1. Submit the qsub scripts under `training/MODEL/_NOTSENSOR/3SENS/` to the SCC cluster or run the Python files directly if you have GPUs available locally.
2. Monitor TensorBoard logs created under `result_qsub/AE_3SENS/` to track training progress.
3. After completion, check the `estimation/Result_3SENS/` folder for updated Excel summaries.

## Summary of the Two Hundred Seventeenth Commit

After running the three-sensor experiments, several analysis notebooks were updated to include both full-sensor and reduced-sensor results in the same plots. The `estimation/compare_sensors.ipynb` notebook now loads the aggregates created in commit 214 and generates side-by-side error curves and bar charts. A short markdown report summarizes which sensor combinations yield the lowest RMSE and impulse metrics.

### Why Combine Results in Shared Visualizations?

Comparing sensors in isolation makes it hard to judge relative performance. By plotting all configurations together, the team could immediately see the trade-offs between model complexity and accuracy. These visual summaries guided decisions on whether the marginal error increase from removing sensors justified the hardware savings.

### Reproducing the Comparison Notebook

1. Ensure the aggregation step from commit 214 has been executed so that summary CSVs exist.
2. Open `estimation/compare_sensors.ipynb` and run all cells.
3. Review the generated figures, which should now include overlays for the full dataset, three-sensor dataset, and any other NOTSENSOR variants.

## Summary of the Two Hundred Eighteenth Commit

The final commit in this series performs a comprehensive cleanup and wraps up the NOTSENSOR investigation. Obsolete intermediate logs were deleted, README references were standardized, and any missing `__init__.py` files were added so helper modules load correctly. The commit message "Applying previous commit" reflects that these fixes finalize the changes introduced over the last several commits.

### Why Conclude with Cleanup?

Long-running research branches accumulate stray files and inconsistencies that can confuse future users. Tidying the repository ensures that cloning at this point provides a working snapshot without manual fixes. It also signals the end of the sensor-ablation phase, paving the way for fresh experiments or a merge back into the main line of development.

### Reproducing the Finalized Repository State

1. Pull the repository at this commit or later to obtain the cleaned directory structure.
2. Install the `CBDtorch` package in editable mode from `training/CBD` as described earlier in the README.
3. Run any of the qsub scripts or notebooks without modification to verify that paths resolve and results are saved in the documented folders.

## Summary of the Two Hundred Nineteenth Commit

This follow-up commit updates the README itself. It captures everything learned while producing the sensorwise aggregation scripts, the specialized three-sensor dataset, the corresponding training jobs, and the concluding cleanup. No source code changed—only these extensive notes were added so future readers can trace the project’s final phase.

### Why Record the Late-Stage History?

With over two hundred revisions, remembering the motivations behind each file becomes difficult. Documenting the closing experiments preserves the reasoning behind the last sensor-ablation tests and ensures the repository can be used as a reference long after active development ends.

### Reproducing the Documentation Update

Simply pull this commit to read the expanded README. All scripts and notebooks continue to run as described earlier.
