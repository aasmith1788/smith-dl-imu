# Step 4: Multi-seed ensemble training with FiLM (Feature-wise Linear Modulation) for demographics
# RESIDUAL VERSION v4: Incorporates demographic data (age, height, weight, sex) via FiLM mechanism
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
NOISE_STD_FRACTION = 0.100  # Optimal noise fraction

# V4: USE DEMOGRAPHICS DATASET
DATASET_NAME = 'IWALQQ_1st_correction_withDemographics'
DATA_TYPE = 'moBWHT'

BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)

# Output directories for ensemble training - V4 FOLDER
BASE_RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results'
RESULTS_DIR = join(BASE_RESULTS_DIR, 'momentpyramidCNN_RESIDUAL_v4_withDemographics')
MODELS_DIR = join(RESULTS_DIR, 'models')
LOGS_DIR = join(RESULTS_DIR, 'logs')
ENSEMBLE_PREDS_DIR = join(RESULTS_DIR, 'ensemble_predictions')

# V4 CONFIGURATION - FiLM with demographics
INITIAL_EXPANSION_CHANNELS = 64  # Expand 42 -> 64 channels before pyramid
N_DEMOGRAPHICS = 4  # age, height, weight, sex
DEMOGRAPHIC_EMBEDDING_DIM = 64  # Shared demographic embedding size

# HYPERPARAMETERS (same as v3)
BEST_PARAMS = {
    "kernels": [3, 7],
    "channels": [24, 48, 96, 192],
    "n_layers": 4,
    "pooling": "adaptive_max",
    "dropout_conv": 0.026340240575068567,
    "dropout_fc": 0.0024323948576126905,
    "lr": 0.0047888970824199635,
    "optimizer": "adamw",
    "loss": "huber",
    "batch_size": 12,
    "weight_decay": 3.2025747173505635e-05,
    "scheduler": "none",
    "grad_clip": 2.6509357919146987,
    "num_heads": 4,
    "noise_frac": 0.100,
    "initial_expansion": INITIAL_EXPANSION_CHANNELS,
    "demographic_embedding_dim": DEMOGRAPHIC_EMBEDDING_DIM
}

# Training parameters
N_FOLDS = 5
ENSEMBLE_EPOCHS = 15
MAX_PARAMETERS = 2_500_000
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
# Dataset utilities with demographics
# ------------------------------
class CNNDataset(Dataset):
    """Dataset that loads IMU sequences AND demographics with optional Gaussian noise."""

    def __init__(self, data_dir, data_type, sess, fold,
                 apply_noise=False, noise_std_fraction=0.015, scaler=None):
        data_file = join(data_dir, f"{fold}_fold_final_{sess}.npz")
        print(f"Loading data from: {data_file}")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        data = np.load(data_file)
        X = np.squeeze(data[f"final_X_{sess}"]).astype(np.float32)
        Y = np.squeeze(data[f"final_Y_{data_type}_{sess}"]).astype(np.float32)
        DG = data[f"final_DG_{sess}"].astype(np.float32)  # Demographics: (n_samples, 4)

        X = X.reshape(-1, 42, 101)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.DG = torch.from_numpy(DG)

        # Noise augmentation settings
        self.apply_noise = apply_noise
        self.noise_std_fraction = noise_std_fraction
        self.scaler = scaler

        if self.apply_noise and self.scaler is not None:
            self.channel_noise_stds = torch.zeros(42)
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
        print(f"  Demographics shape: {self.DG.shape}")
        self.training = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # Shape: (42, 101)
        y = self.Y[idx]
        dg = self.DG[idx]  # Shape: (4,) - age, height, weight, sex

        # Apply Gaussian noise during training only
        if self.apply_noise and self.training:
            noise = torch.randn_like(x)
            for ch in range(42):
                noise[ch, :] *= self.channel_noise_stds[ch]
            x = x + noise

        return x, y, dg

    def set_training(self, mode):
        """Set training mode for noise application."""
        self.training = mode


# ------------------------------
# Model definition with FiLM modulation
# ------------------------------
class SafeBatchNorm1d(nn.Module):
    """BatchNorm1d that handles single-sample batches safely."""

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, x):
        if self.training and x.size(0) == 1:
            return x
        return self.bn(x)


class FiLMGenerator(nn.Module):
    """
    FiLM Generator: Converts demographics into modulation parameters.
    Input: (batch, n_demographics)
    Output: (batch, embedding_dim)
    """

    def __init__(self, n_demographics, embedding_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_demographics, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, demographics):
        """
        Args:
            demographics: (batch, n_demographics)
        Returns:
            embedding: (batch, embedding_dim)
        """
        return self.network(demographics)


class ResidualBlockWithFiLM(nn.Module):
    """
    Residual block with FiLM modulation.
    Applies: output = gamma * F(x) + beta + skip(x)
    FiLM is applied after BN, before ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0, embedding_dim=64):
        super().__init__()

        # Main path
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.bn = SafeBatchNorm1d(out_channels)

        # FiLM projection: embedding_dim -> (gamma, beta) for out_channels
        self.film_projection = nn.Linear(embedding_dim, 2 * out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        # Initialize FiLM to identity (gamma=1, beta=0)
        nn.init.zeros_(self.film_projection.weight)
        nn.init.constant_(self.film_projection.bias[:out_channels], 1.0)  # gamma = 1
        nn.init.constant_(self.film_projection.bias[out_channels:], 0.0)  # beta = 0

    def forward(self, x, demographic_embedding):
        """
        Args:
            x: (batch, in_channels, time)
            demographic_embedding: (batch, embedding_dim)
        Returns:
            out: (batch, out_channels, time)
        """
        # Main path
        out = self.conv(x)
        out = self.bn(out)

        # Apply FiLM modulation
        film_params = self.film_projection(demographic_embedding)  # (batch, 2*out_channels)
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # Each: (batch, out_channels)

        # Reshape for broadcasting: (batch, out_channels, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # FiLM: F_out = gamma * F + beta
        out = gamma * out + beta

        out = self.relu(out)
        out = self.dropout(out)

        # Add skip connection
        skip = self.skip(x)
        out = out + skip

        return out


class PyramidAttnCNN_v4(nn.Module):
    """
    V4: Multi-scale CNN with FiLM for demographic modulation.

    Architecture:
    1. FiLM Generator: demographics -> embedding
    2. Initial expansion layer: 42 -> 64 channels
    3. Multi-branch pyramid with FiLM-modulated residual blocks
    4. Attention mechanism
    5. Output heads
    """

    def __init__(self, kernels, channels, n_layers,
                 dropout_conv, dropout_fc, num_heads=4,
                 num_features=42, num_outputs=303,
                 initial_expansion=64, n_demographics=4,
                 demographic_embedding_dim=64):
        super().__init__()

        # FiLM Generator
        self.film_generator = FiLMGenerator(n_demographics, demographic_embedding_dim)

        # Initial feature expansion (42 -> 64)
        self.initial_expansion = nn.Sequential(
            nn.Conv1d(num_features, initial_expansion, kernel_size=1),
            SafeBatchNorm1d(initial_expansion),
            nn.ReLU()
        )

        # Build pyramid branches with FiLM-modulated residual blocks
        self.branches = nn.ModuleList()

        for k in kernels:
            layers = []
            in_ch = initial_expansion

            for i in range(n_layers):
                out_ch = channels[min(i, len(channels) - 1)]
                layers.append(ResidualBlockWithFiLM(in_ch, out_ch, kernel_size=k,
                                                     dropout=dropout_conv,
                                                     embedding_dim=demographic_embedding_dim))
                in_ch = out_ch

            # Strided convolutions for temporal pyramid
            layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=4, padding=1))
            layers.append(SafeBatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=3, padding=1))
            layers.append(SafeBatchNorm1d(out_ch))
            layers.append(nn.ReLU())

            self.branches.append(nn.ModuleList(layers))

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

    def forward(self, x, demographics):
        """
        Args:
            x: (batch, 42, 101) - IMU signals
            demographics: (batch, 4) - age, height, weight, sex
        Returns:
            output: (batch, 303) - predicted moments
        """
        # Generate demographic embedding
        demographic_embedding = self.film_generator(demographics)  # (batch, embedding_dim)

        # Initial expansion
        x = self.initial_expansion(x)

        # Process through pyramid branches with FiLM
        branch_feats = []
        for branch in self.branches:
            feat = x
            for layer in branch:
                if isinstance(layer, ResidualBlockWithFiLM):
                    feat = layer(feat, demographic_embedding)
                else:
                    feat = layer(feat)
            branch_feats.append(feat)

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
    tmp_model = PyramidAttnCNN_v4(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc'],
        num_heads=BEST_PARAMS['num_heads'],
        initial_expansion=BEST_PARAMS['initial_expansion'],
        n_demographics=N_DEMOGRAPHICS,
        demographic_embedding_dim=BEST_PARAMS['demographic_embedding_dim']
    )
    param_count = count_parameters(tmp_model)
    print(f"Model parameters (v4 with FiLM): {param_count:,}")
    assert param_count < MAX_PARAMETERS, f"Model too large: {param_count:,} parameters"


class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))


# ------------------------------
# Metrics
# ------------------------------
def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler=None):
    """nRMSE (%) for a specific axis in normalized 0-1 space."""
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_dict[axis]

    pred_axis = pred.view(-1, 3, 101)[:, idx, :]
    targ_axis = target.view(-1, 3, 101)[:, idx, :]

    rmse = torch.sqrt(torch.mean((pred_axis - targ_axis) ** 2, dim=1))
    nrmse = 100 * torch.mean(rmse)
    return nrmse.item()


def ensemble_evaluation(models, loader, device, scaler, set_name="TEST",
                       training_ranges=None, epoch=None, print_results=True,
                       return_predictions=False):
    """Comprehensive evaluation for ensemble of models."""
    for model in models:
        model.eval()

    if training_ranges is None:
        training_ranges = {'X': 1.0, 'Y': 1.0, 'Z': 1.0}

    all_ensemble_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, dg in loader:
            X, y, dg = X.to(device), y.to(device), dg.to(device)

            batch_preds = []
            for model in models:
                pred = model(X, dg)
                batch_preds.append(pred.cpu().numpy())

            ensemble_pred = np.mean(batch_preds, axis=0)
            all_ensemble_preds.append(ensemble_pred)
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_ensemble_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if return_predictions:
        return all_preds, all_targets

    pred_X = all_preds[:, 0:101]
    pred_Y = all_preds[:, 101:202]
    pred_Z = all_preds[:, 202:303]

    true_X = all_targets[:, 0:101]
    true_Y = all_targets[:, 101:202]
    true_Z = all_targets[:, 202:303]

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

        print("\nTraining data ranges (normalization denominators):")
        print("-"*70)
        for axis in ['X', 'Y', 'Z']:
            range_val = axis_data[axis]['range']
            print(f"{axis}: range={range_val:.2f} (normalised)")
        print("-"*70)
        print()

    axis_results = []

    for axis in ['X', 'Y', 'Z']:
        pred_axis = axis_data[axis]['pred']
        true_axis = axis_data[axis]['true']
        train_range = axis_data[axis]['range']

        valid_mask = ~(np.isnan(pred_axis).any(axis=1) | np.isnan(true_axis).any(axis=1))
        pred_valid = pred_axis[valid_mask]
        true_valid = true_axis[valid_mask]
        n_valid = valid_mask.sum()
        n_total = len(valid_mask)
        n_dropped = n_total - n_valid

        per_trial_corrs = []
        per_trial_rmses = []
        per_trial_nrmses = []

        for i in range(len(pred_valid)):
            if np.isnan(pred_valid[i]).any() or np.isnan(true_valid[i]).any():
                continue

            corr = np.corrcoef(pred_valid[i], true_valid[i])[0, 1]
            rmse = np.sqrt(np.mean((pred_valid[i] - true_valid[i]) ** 2))
            nrmse = 100 * rmse

            per_trial_corrs.append(corr)
            per_trial_rmses.append(rmse)
            per_trial_nrmses.append(nrmse)

        per_trial_corrs = np.array(per_trial_corrs)
        per_trial_rmses = np.array(per_trial_rmses)
        per_trial_nrmses = np.array(per_trial_nrmses)

        pred_flat = pred_valid.flatten()
        true_flat = true_valid.flatten()

        mask = ~np.isnan(pred_flat) & ~np.isnan(true_flat)
        pred_flat = pred_flat[mask]
        true_flat = true_flat[mask]

        if len(pred_flat) > 1:
            global_corr = np.corrcoef(pred_flat, true_flat)[0, 1]
        else:
            global_corr = np.nan

        global_rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
        global_nrmse = 100 * global_rmse

        true_sd_time = np.nanstd(true_valid, axis=0)
        pred_sd_time = np.nanstd(pred_valid, axis=0)

        valid_timepoints = true_sd_time > 1e-8
        if valid_timepoints.any():
            sd_ratio = np.mean(pred_sd_time[valid_timepoints] / true_sd_time[valid_timepoints])
        else:
            sd_ratio = 0.0

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
# Training utilities
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
    """Train one epoch with demographics."""
    model.train()

    if hasattr(loader.dataset, 'set_training'):
        loader.dataset.set_training(True)

    total_loss = 0
    x_err = y_err = z_err = 0
    for X, y, dg in loader:
        X, y, dg = X.to(device), y.to(device), dg.to(device)
        optimizer.zero_grad()
        out = model(X, dg)
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

    if hasattr(loader.dataset, 'set_training'):
        loader.dataset.set_training(False)

    n = len(loader.dataset)
    return (total_loss / n, x_err / n, y_err / n, z_err / n)


def check_data_availability():
    """Check if required data files exist."""
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

    set_seed(seed)

    model = PyramidAttnCNN_v4(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc'],
        num_heads=BEST_PARAMS['num_heads'],
        initial_expansion=BEST_PARAMS['initial_expansion'],
        n_demographics=N_DEMOGRAPHICS,
        demographic_embedding_dim=BEST_PARAMS['demographic_embedding_dim']
    ).to(device)

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

    log_dir = join(log_dir_base, f'seed_{seed}')
    os.makedirs(log_dir, exist_ok=True)
    writer_train = SummaryWriter(join(log_dir, 'train'))
    writer_test = SummaryWriter(join(log_dir, 'test'))

    for epoch in range(ENSEMBLE_EPOCHS):
        tr_loss, tr_x, tr_y, tr_z = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, BEST_PARAMS['grad_clip']
        )

        writer_train.add_scalar('loss', tr_loss, epoch)
        writer_train.add_scalar('nrmse_mean', (tr_x + tr_y + tr_z) / 3, epoch)

        if epoch % 5 == 4:
            sd_ratio = ensemble_evaluation(
                [model], eval_loader, device, scaler,
                set_name="TRAIN", print_results=False
            )['summary']['avg_sd_ratio']
            writer_train.add_scalar('sd_ratio', sd_ratio, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={tr_loss:.4f}, "
                  f"nRMSE mean={(tr_x + tr_y + tr_z) / 3:.2f}%")

    writer_train.close()
    writer_test.close()

    return model


def multi_seed_ensemble_training():
    """Multi-seed ensemble training with FiLM for demographics - V4."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    check_data_availability()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(ENSEMBLE_PREDS_DIR, exist_ok=True)

    print("="*80)
    print(f"RESIDUAL VERSION v4: WITH FiLM FOR DEMOGRAPHICS")
    print("="*80)
    print(f"Training {N_SEEDS} models per fold with different random seeds")
    print(f"Seeds: {ENSEMBLE_SEEDS}")
    print(f"Training for {ENSEMBLE_EPOCHS} epochs per model")
    print(f"Demographics: age, height, weight, sex (4 features)")
    print(f"FiLM embedding dimension: {DEMOGRAPHIC_EMBEDDING_DIM}")
    print(f"Gaussian noise augmentation: {'ENABLED' if GAUSSIAN_NOISE_ENABLED else 'DISABLED'}")
    if GAUSSIAN_NOISE_ENABLED:
        print(f"  Noise STD fraction: {NOISE_STD_FRACTION:.3f}")
    print(f"ARCHITECTURE: PyramidAttnCNN_v4 with FiLM modulation")
    print(f"Hyperparameters: {BEST_PARAMS}")
    print("="*80)

    config = {
        'timestamp': timestamp,
        'version': 'residual_v4_with_demographics',
        'architecture_change': 'Added FiLM (Feature-wise Linear Modulation) for demographic conditioning',
        'demographics': ['age', 'height', 'weight', 'sex'],
        'n_demographics': N_DEMOGRAPHICS,
        'demographic_embedding_dim': DEMOGRAPHIC_EMBEDDING_DIM,
        'step': 'multi_seed_ensemble_with_FiLM',
        'n_seeds': N_SEEDS,
        'seeds': ENSEMBLE_SEEDS,
        'epochs': ENSEMBLE_EPOCHS,
        'gaussian_noise': {
            'enabled': GAUSSIAN_NOISE_ENABLED,
            'std_fraction': NOISE_STD_FRACTION
        },
        'best_params': BEST_PARAMS,
        'data_split': 'full_train_no_validation'
    }

    with open(join(RESULTS_DIR, f'ensemble_config_{timestamp}.json'), 'w') as f:
        json.dump(config, f, indent=2, allow_nan=False)

    all_fold_results = {}

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold} - RESIDUAL v4 WITH FiLM")
        print(f"{'='*60}")

        scaler_path = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
        scaler = load(open(scaler_path, 'rb'))

        full_train_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'train', fold,
            apply_noise=GAUSSIAN_NOISE_ENABLED,
            noise_std_fraction=BEST_PARAMS['noise_frac'],
            scaler=scaler
        )

        test_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'test', fold,
            apply_noise=False,
            scaler=scaler
        )

        print(f"Full training set: {len(full_train_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")

        train_loader = DataLoader(full_train_dataset,
                                  batch_size=BEST_PARAMS['batch_size'],
                                  shuffle=True, drop_last=True)
        eval_train_loader = DataLoader(full_train_dataset,
                                       batch_size=BEST_PARAMS['batch_size'],
                                       shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=BEST_PARAMS['batch_size'],
                                 shuffle=False, drop_last=False)

        ensemble_models = []
        log_dir_base = join(LOGS_DIR, f'fold_{fold}')

        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            model = train_single_model(
                fold, seed_idx, seed, train_loader, test_loader,
                scaler, log_dir_base
            )
            ensemble_models.append(model)

            model_path = join(MODELS_DIR, f'fold_{fold}_seed_{seed}_model.pt')
            torch.save(model.state_dict(), model_path)

        print(f"\n{'='*60}")
        print(f"ENSEMBLE EVALUATION - FOLD {fold}")
        print(f"{'='*60}")

        full_train_dataset.set_training(False)

        train_results = ensemble_evaluation(
            ensemble_models, eval_train_loader, device, scaler,
            set_name="ENSEMBLE TRAINING", print_results=True
        )

        test_results = ensemble_evaluation(
            ensemble_models, test_loader, device, scaler,
            set_name="ENSEMBLE TEST", print_results=True
        )

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

        fold_results = {
            'fold': fold,
            'n_ensemble_members': N_SEEDS,
            'seeds': ENSEMBLE_SEEDS,
            'architecture': 'PyramidAttnCNN_v4_with_FiLM',
            'demographics': {
                'features': ['age', 'height', 'weight', 'sex'],
                'n_features': N_DEMOGRAPHICS,
                'embedding_dim': DEMOGRAPHIC_EMBEDDING_DIM
            },
            'gaussian_noise': {
                'enabled': GAUSSIAN_NOISE_ENABLED,
                'std_fraction': BEST_PARAMS['noise_frac']
            },
            'training': train_results,
            'test': test_results,
            'hyperparameters': BEST_PARAMS
        }

        all_fold_results[f'fold_{fold}'] = fold_results

        results_file = join(RESULTS_DIR, f'fold_{fold}_ensemble_results.json')
        with open(results_file, 'w') as f:
            json.dump(convert_numpy(fold_results), f, indent=2, allow_nan=False)

        ensemble_path = join(MODELS_DIR, f'fold_{fold}_ensemble_models.pt')
        torch.save({
            'models': [model.state_dict() for model in ensemble_models],
            'config': BEST_PARAMS,
            'seeds': ENSEMBLE_SEEDS,
            'architecture': 'PyramidAttnCNN_v4_with_FiLM',
            'demographics': {
                'n_features': N_DEMOGRAPHICS,
                'embedding_dim': DEMOGRAPHIC_EMBEDDING_DIM
            }
        }, ensemble_path)

    print("\n" + "="*80)
    print("RESIDUAL v4 WITH FiLM TRAINING COMPLETED - SUMMARY ACROSS ALL FOLDS")
    print("="*80)

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

    corr_gap = avg_train_corr - avg_test_corr
    nrmse_gap = avg_test_nrmse - avg_train_nrmse

    print(f"\nGeneralization gaps:")
    print(f"  Correlation gap: {corr_gap:.3f} (train - test)")
    print(f"  nRMSE gap: {nrmse_gap:.2f}% (test - train)")

    if corr_gap < 0.05 and nrmse_gap < 2.0:
        print("✅ Excellent generalization")
    elif corr_gap < 0.10 and nrmse_gap < 5.0:
        print("⚠️  Good generalization")
    else:
        print("❌ Poor generalization - consider more regularization")

    summary = {
        'timestamp': timestamp,
        'version': 'residual_v4_with_demographics',
        'architecture': 'PyramidAttnCNN_v4_with_FiLM',
        'demographics': {
            'features': ['age', 'height', 'weight', 'sex'],
            'n_features': N_DEMOGRAPHICS,
            'embedding_dim': DEMOGRAPHIC_EMBEDDING_DIM,
            'mechanism': 'FiLM (Feature-wise Linear Modulation)'
        },
        'n_folds': N_FOLDS,
        'n_seeds_per_fold': N_SEEDS,
        'seeds': ENSEMBLE_SEEDS,
        'epochs': ENSEMBLE_EPOCHS,
        'optimal_hyperparameters': BEST_PARAMS,
        'gaussian_noise': {
            'enabled': GAUSSIAN_NOISE_ENABLED,
            'std_fraction': BEST_PARAMS['noise_frac']
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
    print("RESIDUAL v4 WITH FiLM TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")
    print(f"Predictions saved to: {ENSEMBLE_PREDS_DIR}")
    print("\nV4 configuration:")
    print(f"- Model: PyramidAttnCNN_v4 with FiLM for demographics")
    print(f"- Demographics: age, height, weight, sex (4 features)")
    print(f"- FiLM: Modulates feature maps via gamma (scale) and beta (shift)")
    print(f"- Architecture: 4 layers with {BEST_PARAMS['channels']} channels")
    print(f"- Initial expansion: 42 -> {INITIAL_EXPANSION_CHANNELS} channels")
    print(f"- Learning rate: {BEST_PARAMS['lr']:.6f}")
    print(f"- Batch size: {BEST_PARAMS['batch_size']}")
    print("\nKey benefits of v4:")
    print("1. FiLM allows demographics to modulate feature extraction dynamically")
    print("2. Demographics act as control signals, not concatenated features")
    print("3. Network learns age/sex-specific biomechanical patterns")
    print("4. Maintains all v3 benefits (initial expansion, residual connections)")


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
