# Step 3: Multi-seed ensemble training for IMU joint angle prediction with Gaussian noise augmentation
# RESIDUAL VERSION v3: Adds initial feature expansion layer (42 -> 64 channels) BEFORE pyramid branches
# Trains multiple models with different seeds and averages predictions
# Adds Gaussian noise to inputs during training to reduce overfitting

import os
import json
import datetime
from os.path import join
import random
import sys
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from pickle import load


# ------------------------------
# Configuration - ENSEMBLE PARAMETERS
# ------------------------------
N_SEEDS = 3  # Number of ensemble members per fold
ENSEMBLE_SEEDS = [42, 123, 456]  # Different seeds for ensemble members

# GAUSSIAN NOISE PARAMETERS
GAUSSIAN_NOISE_ENABLED = True  # Toggle noise augmentation
NOISE_STD_FRACTION = 0.075  # Optimal noise fraction for angles

DATASET_NAME = 'IWALQQ_1st_correction'
DATA_TYPE = 'angle'

BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)

# Output directories for ensemble training - V3 FOLDER
BASE_RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results'
RESULTS_DIR = join(BASE_RESULTS_DIR, 'anglePyramidCNN_RESIDUAL_v3')
MODELS_DIR = join(RESULTS_DIR, 'models')
LOGS_DIR = join(RESULTS_DIR, 'logs')
ENSEMBLE_PREDS_DIR = join(RESULTS_DIR, 'ensemble_predictions')

# V3 CONFIGURATION - Added initial expansion layer
INITIAL_EXPANSION_CHANNELS = 64  # Expand 42 -> 64 channels before pyramid

# HYPERPARAMETERS FROM OPTUNA SEARCH (angle-specific)
BEST_PARAMS = {
    "kernels": [3, 7],
    "channels": [32, 64, 128, 256],
    "n_layers": 4,
    "pooling": "adaptive_max",
    "dropout_conv": 0.04638665342235904,
    "dropout_fc": 0.002461949337930952,
    "lr": 0.005741958295145911,
    "optimizer": "adamw",
    "loss": "huber",
    "batch_size": 16,
    "weight_decay": 9.179123376654393e-05,
    "scheduler": "none",
    "patience": 12,
    "grad_clip": 2.6509357919146987,
    "num_heads": 4,
    "noise_frac": 0.075,
    "initial_expansion": INITIAL_EXPANSION_CHANNELS  # V3: Initial expansion channels
}

# Training parameters
N_FOLDS = 5
ENSEMBLE_EPOCHS = 15
MAX_PARAMETERS = 2_500_000  # Max allowed parameters for model
axis_weights = [1.0, 1.2, 1.2]

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Dataset directory: {DATASET_DIR}")


# ------------------------------
# Reproducibility
# ------------------------------
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Dataset utilities with Gaussian noise
# ------------------------------
class CNNDataset(Dataset):
    """Dataset that reshapes flattened IMU sequences to (42, 101) with optional Gaussian noise."""

    def __init__(self, data_dir, data_type, sess, fold,
                 apply_noise=False, noise_std_fraction=0.015, scaler=None):
        data_file = join(data_dir, f"{fold}_fold_final_{sess}.npz")
        print(f"Loading data from: {data_file}")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        data = np.load(data_file)
        X = np.squeeze(data[f"final_X_{sess}"]).astype(np.float32)
        Y = np.squeeze(data[f"final_Y_{data_type}_{sess}"]).astype(np.float32)
        X = X.reshape(-1, 42, 101)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        # Noise augmentation settings
        self.apply_noise = apply_noise
        self.noise_std_fraction = noise_std_fraction
        self.scaler = scaler

        # Calculate channel-wise noise standard deviations if noise is enabled
        if self.apply_noise and self.scaler is not None:
            # Compute noise STD based on the data range for each channel
            # Assuming 42 channels: potentially 7 IMUs x 6 channels (3 acc + 3 gyro)
            self.channel_noise_stds = torch.zeros(42)

            # Use a simple approach: noise_std = fraction * data_std
            for ch in range(42):
                channel_data = self.X[:, ch, :].flatten()
                channel_std = channel_data.std().item()
                self.channel_noise_stds[ch] = noise_std_fraction * channel_std

            print(f"  Noise augmentation enabled with std fraction={noise_std_fraction:.6f}")
            print(f"  Channel noise STDs range: [{self.channel_noise_stds.min():.4f}, {self.channel_noise_stds.max():.4f}]")

        # Load participant IDs if available
        pid_key = f"participant_{sess}"
        if pid_key in data:
            self.participants = data[pid_key]
        elif f"PID_{sess}" in data:
            self.participants = data[f"PID_{sess}"]
        else:
            self.participants = np.arange(len(X))

        print(f"Loaded {len(self.X)} samples for {sess} set, fold {fold}")
        self.training = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # Shape: (42, 101)
        y = self.Y[idx]

        # Apply Gaussian noise during training only
        if self.apply_noise and self.training:
            # Generate noise with channel-specific standard deviations
            noise = torch.randn_like(x)  # Shape: (42, 101)

            # Scale noise by channel-specific STDs
            for ch in range(42):
                noise[ch, :] *= self.channel_noise_stds[ch]

            # Add noise to input
            x = x + noise

        return x, y

    def set_training(self, mode):
        """Set training mode for noise application."""
        self.training = mode


# ------------------------------
# Model definition with BatchNorm safety and Residual Blocks
# ------------------------------
class SafeBatchNorm1d(nn.Module):
    """BatchNorm1d that handles single-sample batches safely for Conv layers."""

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, x):
        if self.training and x.size(0) == 1:
            return x
        return self.bn(x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for 1D convolutions.
    Implements: output = F(x) + x, where F(x) is Conv->BN->ReLU->Dropout
    Uses 1x1 convolution for skip connection when channels don't match.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()

        # Main path: Conv -> BN -> ReLU -> Dropout
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.bn = SafeBatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection: use 1x1 conv if channels don't match, else identity
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Add skip connection
        skip = self.skip(x)
        out = out + skip

        return out


class PyramidAttnCNN_v3(nn.Module):
    """
    V3: Multi-scale CNN with INITIAL FEATURE EXPANSION, residual connections, and attention.

    Architecture:
    1. Initial expansion layer: 42 -> 64 channels (1x1 conv + BN + ReLU)
    2. Multi-branch pyramid with residual blocks (same as v2)
    3. Attention mechanism
    4. Output heads
    """

    def __init__(self, kernels, channels, n_layers,
                 dropout_conv, dropout_fc, num_heads=4,
                 num_features=42, num_outputs=303,
                 initial_expansion=64):
        super().__init__()

        # V3: Initial feature expansion layer (42 -> 64 channels)
        self.initial_expansion = nn.Sequential(
            nn.Conv1d(num_features, initial_expansion, kernel_size=1),
            SafeBatchNorm1d(initial_expansion),
            nn.ReLU()
        )

        # Now build pyramid branches starting from expanded features
        self.branches = nn.ModuleList()

        for k in kernels:
            layers = []
            in_ch = initial_expansion  # Start from expanded features, not original 42

            # Build residual blocks for each layer
            for i in range(n_layers):
                out_ch = channels[min(i, len(channels) - 1)]
                layers.append(ResidualBlock(in_ch, out_ch, kernel_size=k,
                                            dropout=dropout_conv))
                in_ch = out_ch

            # Keep strided convolutions unchanged (temporal pyramid)
            layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=4,
                                    padding=1))
            layers.append(SafeBatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=3,
                                    padding=1))
            layers.append(SafeBatchNorm1d(out_ch))
            layers.append(nn.ReLU())

            self.branches.append(nn.Sequential(*layers))

        embed_dim = channels[min(n_layers - 1, len(channels) - 1)] * len(kernels)
        self.reduce = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1),
            SafeBatchNorm1d(embed_dim // 2),
            nn.ReLU(),
        )
        attn_dim = embed_dim // 2
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads, batch_first=True)
        self.dropout_attn = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            'X': nn.Linear(attn_dim, 101),
            'Y': nn.Linear(attn_dim, 101),
            'Z': nn.Linear(attn_dim, 101),
        })

    def forward(self, x):
        # V3: First expand features from 42 -> 64
        x = self.initial_expansion(x)

        # Then process through pyramid branches
        branch_feats = [br(x) for br in self.branches]
        x = torch.cat(branch_feats, dim=1)
        x = self.reduce(x)
        seq = x.permute(0, 2, 1)
        seq, _ = self.attn(seq, seq, seq)
        seq = self.dropout_attn(seq)
        pooled = seq.mean(dim=1)
        out_x = self.heads['X'](pooled)
        out_y = self.heads['Y'](pooled)
        out_z = self.heads['Z'](pooled)
        return torch.cat([out_x, out_y, out_z], dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Parameter check
if __name__ == '__main__' and '--skip_param_check' not in sys.argv:
    tmp_model = PyramidAttnCNN_v3(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc'],
        num_heads=BEST_PARAMS['num_heads'],
        initial_expansion=BEST_PARAMS['initial_expansion']
    )
    param_count = count_parameters(tmp_model)
    print(f"Model parameters (v3 - initial expansion to {INITIAL_EXPANSION_CHANNELS} channels): {param_count:,}")
    assert param_count < MAX_PARAMETERS, f"Model too large: {param_count:,} parameters"


class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))


# ------------------------------
# Metrics
# ------------------------------
def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler=None):
    """
    nRMSE (%) for a specific axis in already-normalised 0-1 space.
    Range is 1.0, so nRMSE = 100·RMSE.
    """
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_dict[axis]

    # pred & target are (batch, 303)   ->  reshape to (batch, 3, 101)
    pred_axis = pred.view(-1, 3, 101)[:, idx, :]
    targ_axis = target.view(-1, 3, 101)[:, idx, :]

    rmse  = torch.sqrt(torch.mean((pred_axis - targ_axis) ** 2, dim=1))
    nrmse = 100 * torch.mean(rmse)       # denominator = 1.0
    return nrmse.item()


def ensemble_evaluation(models, loader, device, scaler, set_name="TEST",
                        training_ranges=None, epoch=None, print_results=True,
                        return_predictions=False):
    """
    Comprehensive evaluation for ensemble of models.
    Averages predictions from all models before computing metrics.
    """
    # Put all models in eval mode
    for model in models:
        model.eval()

    # Use the known training ranges
    if training_ranges is None:
        training_ranges = {'X': 1.0, 'Y': 1.0, 'Z': 1.0}  # range in 0-1 space

    # Collect all predictions and targets
    all_ensemble_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # Get predictions from each model
            batch_preds = []
            for model in models:
                pred = model(X)
                batch_preds.append(pred.cpu().numpy())

            # Average predictions across ensemble members
            ensemble_pred = np.mean(batch_preds, axis=0)

            all_ensemble_preds.append(ensemble_pred)
            all_targets.append(y.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_ensemble_preds, axis=0)    # Shape: (n_trials, 303)
    all_targets = np.concatenate(all_targets, axis=0)          # Shape: (n_trials, 303)

    # Optionally return raw predictions
    if return_predictions:
        return all_preds, all_targets

    # Reshape to separate axes: (n_trials, 101) per axis
    pred_X = all_preds[:, 0:101]
    pred_Y = all_preds[:, 101:202]
    pred_Z = all_preds[:, 202:303]

    true_X = all_targets[:, 0:101]
    true_Y = all_targets[:, 101:202]
    true_Z = all_targets[:, 202:303]

    # NOTE: predictions & targets remain in 0-1 space - no inverse transform

    axis_data = {
        'X': {'pred': pred_X, 'true': true_X, 'range': training_ranges['X']},
        'Y': {'pred': pred_Y, 'true': true_Y, 'range': training_ranges['Y']},
        'Z': {'pred': pred_Z, 'true': true_Z, 'range': training_ranges['Z']}
    }

    results = {}

    if print_results:
        if epoch is not None:
            print(f"\n{'='*80}")
            print(f"{set_name} SET EVALUATION (ENSEMBLE) - EPOCH {epoch}")
        else:
            print(f"\n{'='*80}")
            print(f"{set_name} SET EVALUATION (ENSEMBLE) (Final)")
        print(f"{'='*80}")
        print(f"Ensemble size: {len(models)} models")
        if GAUSSIAN_NOISE_ENABLED and "TRAIN" in set_name.upper():
            print(f"Gaussian noise augmentation: ENABLED (std fraction={NOISE_STD_FRACTION:.6f})")

        # Print ranges header
        print("\nTraining data ranges (normalization denominators):")
        print("-"*70)
        for axis in ['X', 'Y', 'Z']:
            range_val = axis_data[axis]['range']
            print(f"{axis}: range={range_val:.2f} (normalised)")
        print("-"*70)
        print()

    # Calculate metrics per axis
    axis_results = []

    for axis in ['X', 'Y', 'Z']:
        pred_axis = axis_data[axis]['pred']
        true_axis = axis_data[axis]['true']
        train_range = axis_data[axis]['range']

        # Remove trials with NaNs
        valid_mask = ~(np.isnan(pred_axis).any(axis=1) | np.isnan(true_axis).any(axis=1))
        pred_valid = pred_axis[valid_mask]
        true_valid = true_axis[valid_mask]
        n_valid = valid_mask.sum()
        n_total = len(valid_mask)
        n_dropped = n_total - n_valid

        # Per-trial correlations and RMSE
        per_trial_corrs = []
        per_trial_rmses = []
        per_trial_nrmses = []

        for i in range(len(pred_valid)):
            # Skip if any NaNs remain
            if np.isnan(pred_valid[i]).any() or np.isnan(true_valid[i]).any():
                continue

            corr = np.corrcoef(pred_valid[i], true_valid[i])[0, 1]
            rmse = np.sqrt(np.mean((pred_valid[i] - true_valid[i]) ** 2))
            nrmse = 100 * rmse                         # range = 1.0

            per_trial_corrs.append(corr)
            per_trial_rmses.append(rmse)
            per_trial_nrmses.append(nrmse)

        per_trial_corrs = np.array(per_trial_corrs)
        per_trial_rmses = np.array(per_trial_rmses)
        per_trial_nrmses = np.array(per_trial_nrmses)

        # Global metrics (flatten all trials)
        pred_flat = pred_valid.flatten()
        true_flat = true_valid.flatten()

        # Remove any NaN positions
        mask = ~np.isnan(pred_flat) & ~np.isnan(true_flat)
        pred_flat = pred_flat[mask]
        true_flat = true_flat[mask]

        if len(pred_flat) > 1:
            global_corr = np.corrcoef(pred_flat, true_flat)[0, 1]
        else:
            global_corr = np.nan

        global_rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
        global_nrmse = 100 * global_rmse           # range = 1.0

        # SD ratio calculation
        true_sd_time = np.nanstd(true_valid, axis=0)
        pred_sd_time = np.nanstd(pred_valid, axis=0)

        # Mean SD ratio (avoiding division by zero)
        valid_timepoints = true_sd_time > 1e-8
        if valid_timepoints.any():
            sd_ratio = np.mean(pred_sd_time[valid_timepoints] / true_sd_time[valid_timepoints])
        else:
            sd_ratio = 0.0

        # Store results
        axis_result = {
            'axis': axis,
            'training_range': train_range,
            'per_trial_corr_mean': per_trial_corrs.mean(),
            'per_trial_corr_std': per_trial_corrs.std(),
            'per_trial_rmse_mean': per_trial_rmses.mean(),
            'per_trial_rmse_std': per_trial_rmses.std(),
            'per_trial_nrmse_mean': per_trial_nrmses.mean(),
            'per_trial_nrmse_std': per_trial_nrmses.std(),
            'global_corr': global_corr,
            'global_rmse': global_rmse,
            'global_nrmse': global_nrmse,
            'sd_ratio': sd_ratio,
            'n_valid': n_valid,
            'n_total': n_total,
            'n_dropped': n_dropped
        }

        axis_results.append(axis_result)
        results[axis] = axis_result

        if print_results:
            print(f"Axis {axis}")
            print(f"  Training range: {train_range:.2f} (normalised)")
            print(f"  Per-trial Corr: {per_trial_corrs.mean():.3f}±{per_trial_corrs.std():.3f}")
            print(f"  Global Corr: {global_corr:.3f}")
            print(f"  Global RMSE: {global_rmse:.3f} (0-1 space)")
            print(f"  Global nRMSE: {global_nrmse:.2f}%")
            print(f"  SD Ratio: {sd_ratio:.3f}")
            print(f"  Per-trial nRMSE: {per_trial_nrmses.mean():.2f}±{per_trial_nrmses.std():.2f}%")
            print(f"  Trials: {n_valid}/{n_total}", end="")
            if n_dropped > 0:
                print(f" (dropped {n_dropped} trials with NaNs)")
            else:
                print()

    # Overall summary
    avg_global_corr = np.mean([r['global_corr'] for r in axis_results])
    avg_global_rmse = np.mean([r['global_rmse'] for r in axis_results])
    avg_global_nrmse = np.mean([r['global_nrmse'] for r in axis_results])
    avg_sd_ratio = np.mean([r['sd_ratio'] for r in axis_results])

    if print_results:
        print(f"\n{'='*80}")
        print(f"{set_name} SUMMARY (ENSEMBLE) - " + (f"EPOCH {epoch}" if epoch is not None else "FINAL"))
        print(f"{'='*80}")
        print(f"Average global correlation: {avg_global_corr:.3f}")
        print(f"Average global RMSE: {avg_global_rmse:.3f} (0-1 space)")
        print(f"Average global nRMSE: {avg_global_nrmse:.2f}%")
        print(f"Average SD ratio: {avg_sd_ratio:.3f}")

        if set_name == "TRAIN" or "TRAINING" in set_name:
            if avg_sd_ratio > 0.9:
                print("✅ SD ratio > 0.9: Ensemble captures good variance")
            elif avg_sd_ratio > 0.8:
                print("⚠️  SD ratio 0.8-0.9: Moderate variance capture")
            else:
                print("❌ SD ratio < 0.8: Under-dispersed predictions")

    results['summary'] = {
        'avg_global_corr': avg_global_corr,
        'avg_global_rmse': avg_global_rmse,
        'avg_global_nrmse': avg_global_nrmse,
        'avg_sd_ratio': avg_sd_ratio
    }

    return results


# ------------------------------
# Training utilities with noise-aware dataset
# ------------------------------
def create_optimizer(params, opt_name, lr, weight_decay):
    if opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == 'nadam':
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)


def create_loss(name):
    if name == 'mse':
        return nn.MSELoss()
    if name == 'mae':
        return nn.L1Loss()
    if name == 'rmse':
        return RMSELoss()
    return nn.HuberLoss()


def train_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip):
    """Train one epoch with Gaussian noise applied to inputs."""
    model.train()

    # Enable training mode in dataset to apply noise
    if hasattr(loader.dataset, 'set_training'):
        loader.dataset.set_training(True)

    total_loss = 0
    x_err = y_err = z_err = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        out_x, out_y, out_z = out[:, :101], out[:, 101:202], out[:, 202:]
        y_x, y_y, y_z = y[:, :101], y[:, 101:202], y[:, 202:]
        loss_x = criterion(out_x, y_x)
        loss_y = criterion(out_y, y_y)
        loss_z = criterion(out_z, y_z)
        loss = axis_weights[0] * loss_x + axis_weights[1] * loss_y + axis_weights[2] * loss_z
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        x_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'x') * batch_size
        y_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'y') * batch_size
        z_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'z') * batch_size

    # Disable training mode after epoch
    if hasattr(loader.dataset, 'set_training'):
        loader.dataset.set_training(False)

    n = len(loader.dataset)
    return (total_loss / n, x_err / n, y_err / n, z_err / n)


def check_data_availability():
    """Check if required data files exist before starting training."""
    print("Checking data availability...")
    print(f"Dataset directory: {DATASET_DIR}")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    missing_files = []
    for fold in range(N_FOLDS):
        train_file = join(DATASET_DIR, f"{fold}_fold_final_train.npz")
        test_file = join(DATASET_DIR, f"{fold}_fold_final_test.npz")
        scaler_file = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")

        for file_path in [train_file, test_file, scaler_file]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"  - {file}")
        raise FileNotFoundError("Please ensure all required data files are present.")

    print("✅ All required data files found!")


def train_single_model(fold, seed_idx, seed, train_loader, test_loader,
                       scaler, log_dir_base):
    """Train a single model with a specific seed."""
    print(f"\n  Training ensemble member {seed_idx+1}/{N_SEEDS} (seed={seed})")

    # Set seed for this model
    set_seed(seed)

    # Initialize model with optimal parameters - V3 with initial expansion
    model = PyramidAttnCNN_v3(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc'],
        num_heads=BEST_PARAMS['num_heads'],
        initial_expansion=BEST_PARAMS['initial_expansion']
    ).to(device)

    # Training setup
    criterion = create_loss(BEST_PARAMS['loss'])
    optimizer = create_optimizer(
        model.parameters(),
        BEST_PARAMS['optimizer'],
        BEST_PARAMS['lr'],
        BEST_PARAMS['weight_decay']
    )

    eval_loader = DataLoader(train_loader.dataset,
                             batch_size=BEST_PARAMS['batch_size'],
                             shuffle=False, drop_last=False)

    # Setup logging for this model
    log_dir = join(log_dir_base, f'seed_{seed}')
    os.makedirs(log_dir, exist_ok=True)
    writer_train = SummaryWriter(join(log_dir, 'train'))
    writer_test = SummaryWriter(join(log_dir, 'test'))

    # Training loop
    for epoch in range(ENSEMBLE_EPOCHS):
        # Training step (with noise)
        tr_loss, tr_x, tr_y, tr_z = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, BEST_PARAMS['grad_clip']
        )

        # Log training metrics
        writer_train.add_scalar('loss', tr_loss, epoch)
        writer_train.add_scalar('nrmse_mean', (tr_x + tr_y + tr_z) / 3, epoch)

        if epoch % 5 == 4:
            sd_ratio = ensemble_evaluation(
                [model], eval_loader, device, scaler,
                set_name="TRAIN", print_results=False
            )['summary']['avg_sd_ratio']
            writer_train.add_scalar('sd_ratio', sd_ratio, epoch)

        # Progress update every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={tr_loss:.4f}, "
                  f"nRMSE mean={(tr_x + tr_y + tr_z) / 3:.2f}%")

    # Close writers
    writer_train.close()
    writer_test.close()

    return model


def multi_seed_ensemble_training():
    """Multi-seed ensemble training using optimal hyperparameters with Gaussian noise and residual connections - V3."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    check_data_availability()

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(ENSEMBLE_PREDS_DIR, exist_ok=True)

    print("="*80)
    print(f"RESIDUAL VERSION v3: INITIAL FEATURE EXPANSION (42 -> {INITIAL_EXPANSION_CHANNELS} channels)")
    print("="*80)
    print(f"Training {N_SEEDS} models per fold with different random seeds")
    print(f"Seeds: {ENSEMBLE_SEEDS}")
    print(f"Training for {ENSEMBLE_EPOCHS} epochs per model")
    print(f"Using complete train+val data (no validation split)")
    print(f"Gaussian noise augmentation: {'ENABLED' if GAUSSIAN_NOISE_ENABLED else 'DISABLED'}")
    if GAUSSIAN_NOISE_ENABLED:
        print(f"  Noise STD fraction: {NOISE_STD_FRACTION:.3f}")
    print(f"ARCHITECTURE: PyramidAttnCNN_v3 with INITIAL EXPANSION + RESIDUAL CONNECTIONS")
    print(f"V3 changes: Initial expansion (42 -> {INITIAL_EXPANSION_CHANNELS}), then {BEST_PARAMS['n_layers']} layers with channels {BEST_PARAMS['channels']}")
    print(f"Hyperparameters: {BEST_PARAMS}")
    print("="*80)
    print("NOTE: All metrics and saved predictions are now in 0-1 "
          "normalised space (matches makeEstimationWithPDF notebooks).")
    print("     Expect RMSE ~0.03 -> nRMSE ~3% instead of 0.03%.")

    # Save configuration
    config = {
        'timestamp': timestamp,
        'version': 'residual_v3',
        'architecture_change': f'Added initial feature expansion layer (42 -> {INITIAL_EXPANSION_CHANNELS} channels) before pyramid branches',
        'step': 'multi_seed_ensemble_optimal_parameters_residual_v3',
        'n_seeds': N_SEEDS,
        'seeds': ENSEMBLE_SEEDS,
        'epochs': ENSEMBLE_EPOCHS,
        'gaussian_noise': {
            'enabled': GAUSSIAN_NOISE_ENABLED,
            'std_fraction': NOISE_STD_FRACTION,
            'description': 'Gaussian noise added to inputs during training only'
        },
        'best_params': BEST_PARAMS,
        'data_split': 'full_train_no_validation',
        'note': f'Residual v3: Initial expansion to {INITIAL_EXPANSION_CHANNELS} channels for richer feature representation before hierarchical learning'
    }

    with open(join(RESULTS_DIR, f'ensemble_config_{timestamp}.json'), 'w') as f:
        json.dump(config, f, indent=2, allow_nan=False)

    # Store results for all folds
    all_fold_results = {}

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold} - RESIDUAL v3 ENSEMBLE (initial expansion + {BEST_PARAMS['n_layers']} layers)")
        print(f"{'='*60}")

        # Load scaler
        scaler_path = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
        scaler = load(open(scaler_path, 'rb'))

        # Load FULL training data with optimal noise augmentation
        full_train_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'train', fold,
            apply_noise=GAUSSIAN_NOISE_ENABLED,
            noise_std_fraction=BEST_PARAMS['noise_frac'],
            scaler=scaler
        )

        # Test dataset WITHOUT noise augmentation
        test_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'test', fold,
            apply_noise=False,  # Never apply noise to test set
            scaler=scaler
        )

        print(f"Full training set: {len(full_train_dataset)} samples (with noise augmentation)")
        print(f"Test set: {len(test_dataset)} samples (no noise)")

        # Create data loaders
        train_loader = DataLoader(full_train_dataset,
                                  batch_size=BEST_PARAMS['batch_size'],
                                  shuffle=True, drop_last=True)
        eval_train_loader = DataLoader(full_train_dataset,
                                       batch_size=BEST_PARAMS['batch_size'],
                                       shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=BEST_PARAMS['batch_size'],
                                 shuffle=False, drop_last=False)

        # Train ensemble members
        ensemble_models = []
        log_dir_base = join(LOGS_DIR, f'fold_{fold}')

        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            model = train_single_model(
                fold, seed_idx, seed, train_loader, test_loader,
                scaler, log_dir_base
            )
            ensemble_models.append(model)

            # Save individual model
            model_path = join(MODELS_DIR, f'fold_{fold}_seed_{seed}_model.pt')
            torch.save(model.state_dict(), model_path)

        # Ensemble evaluation
        print(f"\n{'='*60}")
        print(f"ENSEMBLE EVALUATION - FOLD {fold}")
        print(f"{'='*60}")

        # Disable noise for evaluation
        full_train_dataset.set_training(False)

        # Training set evaluation (without noise)
        train_results = ensemble_evaluation(
            ensemble_models, eval_train_loader, device, scaler,
            set_name="ENSEMBLE TRAINING", print_results=True
        )

        # Test set evaluation
        test_results = ensemble_evaluation(
            ensemble_models, test_loader, device, scaler,
            set_name="ENSEMBLE TEST", print_results=True
        )

        # Save ensemble predictions
        print("\nSaving ensemble predictions...")
        train_preds, train_targets = ensemble_evaluation(
            ensemble_models, eval_train_loader, device, scaler,
            return_predictions=True
        )
        test_preds, test_targets = ensemble_evaluation(
            ensemble_models, test_loader, device, scaler,
            return_predictions=True
        )

        pred_file = join(ENSEMBLE_PREDS_DIR, f'fold_{fold}_ensemble_predictions.npz')
        np.savez(pred_file,
                 train_predictions=train_preds,
                 train_targets=train_targets,
                 test_predictions=test_preds,
                 test_targets=test_targets)
        print(f"Saved predictions to: {pred_file}")

        # Save comprehensive results
        fold_results = {
            'fold': fold,
            'n_ensemble_members': N_SEEDS,
            'seeds': ENSEMBLE_SEEDS,
            'architecture': 'PyramidAttnCNN_v3_with_InitialExpansion',
            'gaussian_noise': {
                'enabled': GAUSSIAN_NOISE_ENABLED,
                'std_fraction': BEST_PARAMS['noise_frac'],
                'optimal': True
            },
            'training': train_results,
            'test': test_results,
            'hyperparameters': BEST_PARAMS
        }

        all_fold_results[f'fold_{fold}'] = fold_results

        # Save individual fold results
        results_file = join(RESULTS_DIR, f'fold_{fold}_ensemble_results.json')
        with open(results_file, 'w') as f:
            json.dump(convert_numpy(fold_results), f, indent=2, allow_nan=False)

        # Save entire ensemble (all models) for potential future use
        ensemble_path = join(MODELS_DIR, f'fold_{fold}_ensemble_models.pt')
        torch.save({
            'models': [model.state_dict() for model in ensemble_models],
            'config': BEST_PARAMS,
            'seeds': ENSEMBLE_SEEDS,
            'architecture': 'PyramidAttnCNN_v3_with_InitialExpansion',
            'gaussian_noise': {
                'enabled': GAUSSIAN_NOISE_ENABLED,
                'std_fraction': BEST_PARAMS['noise_frac'],
                'optimal': True
            }
        }, ensemble_path)

    # Summary across all folds
    print("\n" + "="*80)
    print("RESIDUAL v3 ANGLE ENSEMBLE TRAINING COMPLETED - SUMMARY ACROSS ALL FOLDS")
    print("="*80)

    # Calculate average metrics across folds
    avg_train_corr = np.mean([all_fold_results[f'fold_{i}']['training']['summary']['avg_global_corr']
                              for i in range(N_FOLDS)])
    avg_test_corr = np.mean([all_fold_results[f'fold_{i}']['test']['summary']['avg_global_corr']
                             for i in range(N_FOLDS)])
    avg_train_nrmse = np.mean([all_fold_results[f'fold_{i}']['training']['summary']['avg_global_nrmse']
                               for i in range(N_FOLDS)])
    avg_test_nrmse = np.mean([all_fold_results[f'fold_{i}']['test']['summary']['avg_global_nrmse']
                              for i in range(N_FOLDS)])
    avg_train_sd = np.mean([all_fold_results[f'fold_{i}']['training']['summary']['avg_sd_ratio']
                            for i in range(N_FOLDS)])
    avg_test_sd = np.mean([all_fold_results[f'fold_{i}']['test']['summary']['avg_sd_ratio']
                           for i in range(N_FOLDS)])

    print(f"Average across {N_FOLDS} folds:")
    print(f"  Training - Corr: {avg_train_corr:.3f}, nRMSE: {avg_train_nrmse:.2f}%, SD ratio: {avg_train_sd:.3f}")
    print(f"  Test     - Corr: {avg_test_corr:.3f}, nRMSE: {avg_test_nrmse:.2f}%, SD ratio: {avg_test_sd:.3f}")

    # Calculate train-test gap
    corr_gap = avg_train_corr - avg_test_corr
    nrmse_gap = avg_test_nrmse - avg_train_nrmse

    print(f"\nGeneralization gaps:")
    print(f"  Correlation gap: {corr_gap:.3f} (train - test)")
    print(f"  nRMSE gap: {nrmse_gap:.2f}% (test - train)")

    if corr_gap < 0.05 and nrmse_gap < 2.0:
        print("✅ Excellent generalization: minimal train-test gap")
    elif corr_gap < 0.10 and nrmse_gap < 5.0:
        print("⚠️  Good generalization: moderate train-test gap")
    else:
        print("❌ Poor generalization: large train-test gap - consider more regularization")

    # Save overall summary
    summary = {
        'timestamp': timestamp,
        'version': 'residual_v3',
        'architecture': 'PyramidAttnCNN_v3_with_InitialExpansion',
        'n_folds': N_FOLDS,
        'n_seeds_per_fold': N_SEEDS,
        'seeds': ENSEMBLE_SEEDS,
        'epochs': ENSEMBLE_EPOCHS,
        'optimal_hyperparameters': BEST_PARAMS,
        'gaussian_noise': {
            'enabled': GAUSSIAN_NOISE_ENABLED,
            'std_fraction': BEST_PARAMS['noise_frac'],
            'optimal': True,
            'benefit': 'Reduces overfitting by making model robust to small input variations'
        },
        'residual_connections': {
            'enabled': True,
            'benefit': 'Improves gradient flow and enables training of deeper networks'
        },
        'v3_changes': {
            'initial_expansion': INITIAL_EXPANSION_CHANNELS,
            'n_layers': BEST_PARAMS['n_layers'],
            'channels': BEST_PARAMS['channels'],
            'benefit': f'Initial expansion to {INITIAL_EXPANSION_CHANNELS} channels provides richer feature representation before hierarchical learning'
        },
        'average_metrics': {
            'train': {
                'avg_correlation': avg_train_corr,
                'avg_nrmse': avg_train_nrmse,
                'avg_sd_ratio': avg_train_sd
            },
            'test': {
                'avg_correlation': avg_test_corr,
                'avg_nrmse': avg_test_nrmse,
                'avg_sd_ratio': avg_test_sd
            },
            'generalization': {
                'correlation_gap': corr_gap,
                'nrmse_gap': nrmse_gap
            }
        },
        'fold_results': all_fold_results
    }

    summary_file = join(RESULTS_DIR, f'optimal_ensemble_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2, allow_nan=False)

    print(f"\n{'='*80}")
    print("RESIDUAL v3 ANGLE ENSEMBLE TRAINING WITH INITIAL EXPANSION COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")
    print(f"Predictions saved to: {ENSEMBLE_PREDS_DIR}")
    print("\nV3 configuration used:")
    print(f"- Model: PyramidAttnCNN_v3 with INITIAL EXPANSION + RESIDUAL BLOCKS")
    print(f"- Initial expansion: 42 -> {INITIAL_EXPANSION_CHANNELS} channels (1x1 conv + BN + ReLU)")
    print(f"- Architecture: {BEST_PARAMS['n_layers']} layers with {BEST_PARAMS['channels']} channels")
    print(f"- Attention heads: {BEST_PARAMS['num_heads']}")
    print(f"- Kernels: {BEST_PARAMS['kernels']} (multi-scale convolution)")
    print(f"- Learning rate: {BEST_PARAMS['lr']:.6f}")
    print(f"- Batch size: {BEST_PARAMS['batch_size']}")
    print(f"- Dropout: conv={BEST_PARAMS['dropout_conv']:.4f}, fc={BEST_PARAMS['dropout_fc']:.4f}")
    print(f"- Weight decay: {BEST_PARAMS['weight_decay']:.2e}")
    print(f"- Noise augmentation: {BEST_PARAMS['noise_frac']:.3f}")
    print(f"- Training epochs: {ENSEMBLE_EPOCHS} (conservative for generalization)")
    print("\nKey benefits of v3:")
    print(f"1. Initial expansion (42 -> {INITIAL_EXPANSION_CHANNELS}) provides richer feature space before processing")
    print("2. More feature maps than first ResidualBlock for better initial representation")
    print("3. Residual connections improve gradient flow through deeper network")
    print("4. Gaussian noise augmentation reduces overfitting")


def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


if __name__ == '__main__':
    multi_seed_ensemble_training()
