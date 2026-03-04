"""
Patch both estimation notebooks:
1. Replace Cell 'load-ensemble-predictions' with new naive_baseline_evaluate
2. Insert new linear regression baseline cell after the metrics cell
"""
import json, os

MOMENT_NB = 'C:/Users/aasmi/smith-dl-imu/estimation/notsensor/makeEstimationWithPDF_PyramidAttnCNN_RESIDUAL_v3.ipynb'
ANGLE_NB  = 'C:/Users/aasmi/smith-dl-imu/estimation/notsensor/makeEstimationWithPDF_PyramidAttnCNN_RESIDUAL_angles.ipynb'

# ==============================================================================
# CELL SOURCES
# ==============================================================================

def naive_cell_source(use_degrees: bool) -> str:
    unit = "°" if use_degrees else " Nm/BW\u00b7Ht"
    unit_range = f"{{train_range:.2f}}{unit} (min {{train_min:.2f}}{'°' if use_degrees else ''}, max {{train_max:.2f}}{'°' if use_degrees else ''})"
    return f'''import numpy as np


def naive_baseline_evaluate(
    train_X, train_Y, train_Z,
    test_X, test_Y, test_Z,
    timepoints_per_axis=101,
    drop_nan_trials=True,
    eps=1e-8
):
    """
    TRUE NAIVE BASELINE: Predicts a single constant value (overall mean across
    ALL training data - all trials, all timepoints) for every timepoint.
    Produces a flat horizontal line prediction.
    """
    axis_labels = ['X', 'Y', 'Z']
    train_arrays = [train_X, train_Y, train_Z]
    test_arrays  = [test_X,  test_Y,  test_Z]

    if train_X.shape[0] == 0 or test_X.shape[0] == 0:
        print("\\u274c Cannot run naive baseline: No training or test data loaded.")
        return None

    def _maybe_slice(A, axis_index):
        if A.shape[1] == 3 * timepoints_per_axis:
            sl = slice(axis_index * timepoints_per_axis, (axis_index + 1) * timepoints_per_axis)
            return A[:, sl]
        return A

    results = {{}}
    print("=" * 80)
    print("TRUE NAIVE BASELINE ANALYSIS (Constant Mean Baseline)")
    print("=" * 80)
    print("Baseline predicts a SINGLE CONSTANT value (overall training mean)")
    print("for every timepoint - flat horizontal line prediction.")
    print("=" * 80)

    for i, ax in enumerate(axis_labels):
        train_axis = _maybe_slice(train_arrays[i], i)
        test_axis  = _maybe_slice(test_arrays[i],  i)

        overall_train_mean = np.nanmean(train_axis)
        train_min  = np.nanmin(train_axis)
        train_max  = np.nanmax(train_axis)
        train_range = max(train_max - train_min, eps)

        baseline_preds = np.full_like(test_axis, overall_train_mean)

        if drop_nan_trials:
            valid_mask = ~np.isnan(test_axis).any(axis=1) & ~np.isnan(baseline_preds).any(axis=1)
        else:
            valid_mask = np.ones(test_axis.shape[0], dtype=bool)

        test_axis_valid = test_axis[valid_mask]
        baseline_valid  = baseline_preds[valid_mask]

        if test_axis_valid.shape[0] == 0:
            results[ax] = {{"error": "No valid trials"}}
            continue

        flat_true = test_axis_valid.reshape(-1)
        flat_pred = baseline_valid.reshape(-1)
        m = ~np.isnan(flat_true) & ~np.isnan(flat_pred)
        flat_true, flat_pred = flat_true[m], flat_pred[m]

        global_rmse = np.sqrt(np.mean((flat_pred - flat_true) ** 2)) if m.any() else np.nan
        global_corr = (np.corrcoef(flat_true, flat_pred)[0, 1]
                       if flat_true.size > 1 and np.std(flat_true) > eps and np.std(flat_pred) > eps
                       else 0.0)
        global_nrmse_pct = 100 * global_rmse / train_range if not np.isnan(global_rmse) else np.nan

        print(f"\\nAxis {{ax}}")
        print(f"  Constant baseline value: {{overall_train_mean:.4f}}")
        print(f"  Training range: {unit_range}")
        print(f"  Global Corr: {{global_corr:.3f}} | Global nRMSE: {{global_nrmse_pct:.2f}}%")

        results[ax] = {{
            "global_corr":      global_corr,
            "global_rmse":      global_rmse,
            "global_nrmse_pct": global_nrmse_pct,
            "training_range":   train_range,
        }}

    valid_axes = [ax for ax in axis_labels if "error" not in results.get(ax, {{}})]
    if valid_axes:
        avg_corr  = float(np.nanmean([results[a]["global_corr"]      for a in valid_axes]))
        avg_nrmse = float(np.nanmean([results[a]["global_nrmse_pct"] for a in valid_axes]))
        print("\\n" + "=" * 80)
        print("NAIVE BASELINE SUMMARY")
        print("=" * 80)
        print(f"Average global correlation: {{avg_corr:.3f}}")
        print(f"Average global nRMSE: {{avg_nrmse:.2f}}%")
        results["summary"] = {{"avg_corr": avg_corr, "avg_nrmse_pct": avg_nrmse}}

    return results


# --- Run ---
if 'bl_train_X' in locals() and bl_train_X.size > 0:
    naive_results = naive_baseline_evaluate(
        bl_train_X, bl_train_Y, bl_train_Z,
        bl_test_X,  bl_test_Y,  bl_test_Z
    )
else:
    print("\\nSkipping Naive Baseline: input data missing.")
    naive_results = None
'''


LINEAR_CELL = '''\
import json
import os
from os.path import join
from pickle import load as pickle_load

import numpy as np
from sklearn.linear_model import Ridge


def compute_metrics(y_true, y_pred, train_range, eps=1e-8):
    """Global correlation and nRMSE."""
    ft = y_true.reshape(-1)
    fp = y_pred.reshape(-1)
    m  = ~np.isnan(ft) & ~np.isnan(fp)
    ft, fp = ft[m], fp[m]
    if ft.size < 2:
        return 0.0, np.nan
    rmse      = np.sqrt(np.mean((fp - ft) ** 2))
    nrmse_pct = 100 * rmse / train_range if train_range > eps else np.nan
    corr      = (np.corrcoef(ft, fp)[0, 1]
                 if np.std(ft) > eps and np.std(fp) > eps else 0.0)
    return corr, nrmse_pct


def linear_regression_baseline_per_fold(
    data_dir, data_type, num_folds=5, alpha=1.0, output_dir=None, eps=1e-8
):
    """
    Ridge regression baseline evaluated per fold on BOTH train and test.
    Tests whether the CNN's complexity is justified over a linear model.
    """
    axis_labels    = ['X', 'Y', 'Z']
    all_fold_results = []

    print("\\n" + "=" * 80)
    print("LINEAR REGRESSION BASELINE - PER-FOLD EVALUATION")
    print("=" * 80)
    print(f"Dataset : {data_dir}")
    print(f"Type    : {data_type}  |  Ridge alpha: {alpha}")
    print("=" * 80)

    for fold in range(num_folds):
        print(f"\\n{'='*60}\\nFOLD {fold}\\n{'='*60}")
        fold_results = {"fold": fold, "axes": {}}

        train_npz   = join(data_dir, f"{fold}_fold_final_train.npz")
        test_npz    = join(data_dir, f"{fold}_fold_final_test.npz")
        scaler_path = join(data_dir, f"{fold}_fold_scaler4Y_{data_type}.pkl")

        if not os.path.exists(train_npz) or not os.path.exists(test_npz):
            print(f"  \u274c Missing NPZ files for fold {fold}. Skipping.")
            fold_results["error"] = "Missing NPZ files"
            all_fold_results.append(fold_results)
            continue

        try:
            train_data = np.load(train_npz)
            test_data  = np.load(test_npz)
            X_train = train_data["final_X_train"]
            X_test  = test_data["final_X_test"]
            if X_train.ndim == 3: X_train = X_train.reshape(X_train.shape[0], -1)
            if X_test.ndim  == 3: X_test  = X_test.reshape(X_test.shape[0],  -1)

            Y_tr = train_data[f"final_Y_{data_type}_train"]
            Y_te = test_data[f"final_Y_{data_type}_test"]

            with open(scaler_path, 'rb') as fh:
                scaler = pickle_load(fh)

            n_tr, n_te = Y_tr.shape[0], Y_te.shape[0]
            print(f"  Train: {n_tr}  Test: {n_te}  Features: {X_train.shape[1]}")

            Y_tr_r = Y_tr.reshape(n_tr, 3, 101)
            Y_te_r = Y_te.reshape(n_te, 3, 101)

            s_min   = scaler.min_.reshape(1, 3)
            s_scale = scaler.scale_.reshape(1, 3)

            Y_tr_u = np.stack([((Y_tr_r[i].T - s_min) / s_scale).T for i in range(n_tr)])
            Y_te_u = np.stack([((Y_te_r[i].T - s_min) / s_scale).T for i in range(n_te)])

        except Exception as e:
            print(f"  \u274c Error loading fold {fold}: {e}")
            fold_results["error"] = str(e)
            all_fold_results.append(fold_results)
            continue

        f_tr_corrs, f_tr_nrmses, f_te_corrs, f_te_nrmses = [], [], [], []

        for ax_idx, ax in enumerate(axis_labels):
            y_tr_ax = Y_tr_u[:, ax_idx, :]
            y_te_ax = Y_te_u[:, ax_idx, :]
            tr_range = max(np.nanmax(y_tr_ax) - np.nanmin(y_tr_ax), eps)

            tr_ok = ~np.isnan(y_tr_ax).any(axis=1) & ~np.isnan(X_train).any(axis=1)
            te_ok = ~np.isnan(y_te_ax).any(axis=1) & ~np.isnan(X_test).any(axis=1)

            mdl = Ridge(alpha=alpha, fit_intercept=True)
            mdl.fit(X_train[tr_ok], y_tr_ax[tr_ok])

            tr_c, tr_n = compute_metrics(y_tr_ax[tr_ok], mdl.predict(X_train[tr_ok]), tr_range)
            te_c, te_n = compute_metrics(y_te_ax[te_ok], mdl.predict(X_test[te_ok]),  tr_range)

            fold_results["axes"][ax] = {
                "train_corr": float(tr_c), "train_nrmse": float(tr_n),
                "test_corr":  float(te_c), "test_nrmse":  float(te_n),
                "n_train": int(tr_ok.sum()), "n_test": int(te_ok.sum()),
            }
            f_tr_corrs.append(tr_c); f_tr_nrmses.append(tr_n)
            f_te_corrs.append(te_c); f_te_nrmses.append(te_n)

        fold_results.update({
            "avg_train_corr":  float(np.mean(f_tr_corrs)),
            "avg_train_nrmse": float(np.mean(f_tr_nrmses)),
            "avg_test_corr":   float(np.mean(f_te_corrs)),
            "avg_test_nrmse":  float(np.mean(f_te_nrmses)),
            "avg_corr_gap":    float(np.mean(f_tr_corrs) - np.mean(f_te_corrs)),
            "avg_nrmse_gap":   float(np.mean(f_te_nrmses) - np.mean(f_tr_nrmses)),
        })
        all_fold_results.append(fold_results)

        print(f"  {'Axis':<6} {'Train Corr':>12} {'Train nRMSE':>12} {'Test Corr':>12} {'Test nRMSE':>12} {'Gap':>10}")
        print(f"  {'-'*70}")
        for ax in axis_labels:
            r = fold_results["axes"].get(ax, {})
            if "error" not in r:
                print(f"  {ax:<6} {r['train_corr']:>12.4f} {r['train_nrmse']:>11.2f}% "
                      f"{r['test_corr']:>12.4f} {r['test_nrmse']:>11.2f}% "
                      f"{r['train_corr']-r['test_corr']:>10.4f}")
        fr = fold_results
        print(f"  {'AVG':<6} {fr['avg_train_corr']:>12.4f} {fr['avg_train_nrmse']:>11.2f}% "
              f"{fr['avg_test_corr']:>12.4f} {fr['avg_test_nrmse']:>11.2f}% "
              f"{fr['avg_corr_gap']:>10.4f}")

    valid_folds = [f for f in all_fold_results if "avg_train_corr" in f]
    summary = {}
    if valid_folds:
        summary = {
            "num_folds":       len(valid_folds),
            "avg_train_corr":  float(np.mean([f["avg_train_corr"]  for f in valid_folds])),
            "std_train_corr":  float(np.std( [f["avg_train_corr"]  for f in valid_folds])),
            "avg_train_nrmse": float(np.mean([f["avg_train_nrmse"] for f in valid_folds])),
            "avg_test_corr":   float(np.mean([f["avg_test_corr"]   for f in valid_folds])),
            "std_test_corr":   float(np.std( [f["avg_test_corr"]   for f in valid_folds])),
            "avg_test_nrmse":  float(np.mean([f["avg_test_nrmse"]  for f in valid_folds])),
            "avg_corr_gap":    float(np.mean([f["avg_corr_gap"]    for f in valid_folds])),
            "avg_nrmse_gap":   float(np.mean([f["avg_nrmse_gap"]   for f in valid_folds])),
        }

        print("\\n" + "=" * 80)
        print("LINEAR REGRESSION SUMMARY ACROSS ALL FOLDS")
        print("=" * 80)
        print(f"\\n{'Metric':<25} {'Train':>22} {'Test':>22} {'Gap':>10}")
        print("-" * 82)
        print(f"{'Correlation':<25} {summary['avg_train_corr']:.4f} \u00b1 {summary['std_train_corr']:.4f}"
              f"   {summary['avg_test_corr']:.4f} \u00b1 {summary['std_test_corr']:.4f}"
              f"   {summary['avg_corr_gap']:.4f}")
        print(f"{'nRMSE (%)':<25} {summary['avg_train_nrmse']:.2f}%"
              f"   {summary['avg_test_nrmse']:.2f}%"
              f"   {summary['avg_nrmse_gap']:.2f}%")

        gap = summary['avg_corr_gap']
        label = ("\u26a0\ufe0f  OVERFITTING DETECTED" if gap > 0.05
                 else "\U0001f4ca MILD OVERFITTING"   if gap > 0.02
                 else "\u2705 GOOD GENERALIZATION")
        print(f"\\n{label}: train-test gap = {gap:.4f}")
        print(f"Compare CNN test corr against linear baseline: {summary['avg_test_corr']:.3f}")

    out = {"baseline_type": "linear_regression", "alpha": alpha,
           "data_type": data_type, "folds": all_fold_results, "summary": summary}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p = join(output_dir, f"linear_regression_baseline_{data_type}.json")
        with open(p, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f"\\n\U0001f4c1 Saved: {p}")

    return out


# --- Run linear regression baseline ---
linear_results = linear_regression_baseline_per_fold(
    data_dir=DATASET_DIR,
    data_type=DATA_TYPE,
    num_folds=5,
    alpha=1.0,
    output_dir=outputExcelBaseDir,
)

# --- Comparison table ---
print("\\n" + "=" * 80)
print("BASELINE COMPARISON SUMMARY")
print("=" * 80)

naive_corr  = naive_results["summary"]["avg_corr"]      if naive_results and "summary" in naive_results else 0.0
naive_nrmse = naive_results["summary"]["avg_nrmse_pct"] if naive_results and "summary" in naive_results else float('nan')

if linear_results.get("summary"):
    s = linear_results["summary"]
    lr_tr_c, lr_tr_n = s["avg_train_corr"], s["avg_train_nrmse"]
    lr_te_c, lr_te_n = s["avg_test_corr"],  s["avg_test_nrmse"]
else:
    lr_tr_c = lr_tr_n = lr_te_c = lr_te_n = float('nan')

print(f"\\n{'Baseline':<25} {'Train Corr':>12} {'Test Corr':>12} {'Train nRMSE':>12} {'Test nRMSE':>12}")
print("-" * 75)
print(f"{'Naive (Constant Mean)':<25} {'N/A':>12} {naive_corr:>12.3f} {'N/A':>12} {naive_nrmse:>11.2f}%")
print(f"{'Linear Regression':<25} {lr_tr_c:>12.4f} {lr_te_c:>12.4f} {lr_tr_n:>11.2f}% {lr_te_n:>11.2f}%")
print("-" * 75)
print(f"\\nLinear vs Naive (Test): Corr +{lr_te_c - naive_corr:.3f},  nRMSE -{naive_nrmse - lr_te_n:.2f}pp")
'''


def new_cell(source, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source}


def patch_notebook(nb_path, use_degrees):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # 1. Replace Cell 'load-ensemble-predictions'
    naive_src = naive_cell_source(use_degrees)
    replaced = False
    last_metrics_idx = None
    for i, cell in enumerate(cells):
        if cell.get('id') == 'load-ensemble-predictions':
            cell['source'] = naive_src
            cell['outputs'] = []
            cell['execution_count'] = None
            replaced = True
            print(f"  Replaced naive baseline cell (idx {i})")
        # Track last metrics cell to insert after
        src = ''.join(cell.get('source', []))
        if 'main_metrics' in src or 'evaluate_axis_metrics' in src or 'CALCULATING FINAL TRAINING' in src:
            last_metrics_idx = i

    if not replaced:
        print("  WARNING: 'load-ensemble-predictions' cell not found!")

    # 2. Insert linear regression cell after last metrics cell (or at end)
    insert_after = last_metrics_idx if last_metrics_idx is not None else len(cells) - 1
    lin_id = 'linear-regression-baseline'
    # Remove existing linear regression cell if present
    cells = [c for c in cells if c.get('id') != lin_id]
    cells.insert(insert_after + 1, new_cell(LINEAR_CELL, lin_id))
    print(f"  Inserted linear regression cell after idx {insert_after}")

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved: {nb_path}")


print("Patching moment notebook...")
patch_notebook(MOMENT_NB, use_degrees=False)

print("\nPatching angle notebook...")
patch_notebook(ANGLE_NB, use_degrees=True)

print("\nDone.")
