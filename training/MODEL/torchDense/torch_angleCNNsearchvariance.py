# Step 3b: Multi-seed ensemble training with Gaussian noise + variance-encouragement loss
# Addresses low SD-ratio by adding penalty when prediction spread falls below target spread
# Trains multiple models with different seeds and averages predictions

import os
import json
import datetime
from os.path import join
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from pickle import load
from sklearn.model_selection import GroupShuffleSplit


# ------------------------------
# Configuration - ENSEMBLE PARAMETERS WITH VARIANCE ENCOURAGEMENT
# ------------------------------
N_SEEDS = 3  # Number of ensemble members per fold
ENSEMBLE_SEEDS = [42, 123, 456]  # Different seeds for ensemble members

# GAUSSIAN NOISE PARAMETERS
GAUSSIAN_NOISE_ENABLED = True  # Toggle noise augmentation
NOISE_STD_FRACTION = 0.075

# VARIANCE ENCOURAGEMENT PARAMETERS
VARIANCE_LOSS_ENABLED = True  # Toggle variance encouragement loss
VARIANCE_LOSS_COEFFICIENT = 0.03  # 3% of base loss (range: 0.02-0.05)
VARIANCE_THRESHOLD = 0.90  # Penalty activates when pred_spread < 90% of target_spread
MIN_BATCH_SIZE_FOR_VARIANCE = 4  # Minimum batch size to compute variance loss

DATASET_NAME = 'IWALQQ_1st_correction'
DATA_TYPE = 'angle'

BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)

# Output directories for ensemble training WITH VARIANCE ENCOURAGEMENT
RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\MultiScaleCNN_ensemble_variance_loss'
MODELS_DIR = join(RESULTS_DIR, 'models')
LOGS_DIR = join(RESULTS_DIR, 'logs')
ENSEMBLE_PREDS_DIR = join(RESULTS_DIR, 'ensemble_predictions')

# Directories for searching variance loss coefficient
VARIANCE_SEARCH_RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\MultiScaleCNN_variance_coefficient_search'
VARIANCE_SEARCH_MODELS_DIR = join(VARIANCE_SEARCH_RESULTS_DIR, 'models')
VARIANCE_SEARCH_LOGS_DIR = join(VARIANCE_SEARCH_RESULTS_DIR, 'logs')
VARIANCE_SEARCH_DATA_DIR = join(VARIANCE_SEARCH_RESULTS_DIR, 'search_results')

# OPTIMAL HYPERPARAMETERS FROM OPTUNA SEARCH
BEST_PARAMS = {
    "kernels": [3, 7],
    "channels": [32, 64, 128],
    "n_layers": 3,
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
    "grad_clip": 2.6509357919146987
}

# Training parameters
N_FOLDS = 5 # 🧪 TEST MODE: Start with 1 fold for pipeline verification
ENSEMBLE_EPOCHS = 15
MAX_PARAMETERS = 500_000

# Hyperparameter search configuration
VARIANCE_COEFFS_TO_TEST = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
SEARCH_SEEDS = [42, 123]
SEARCH_EPOCHS = 15

# 🧪 TEST MODE: Start with minimal search for pipeline verification
TEST_MODE = False

if TEST_MODE:
    VARIANCE_COEFFS_TO_TEST = [0.0]  # Just test baseline first
    N_FOLDS = 1  # Test one fold only
    print("🧪 RUNNING IN TEST MODE: Testing coefficient 0.0 on fold 0 only")
    print("   Change TEST_MODE = False for full search")

# 🧪 TEST MODE: Single seed for quick validation
#_SEEDS = 1  # Test with 1 seed first to verify variance loss effectiveness
ENSEMBLE_SEEDS = [42]  # Single seed for testing

# 📝 PRODUCTION MODE: Uncomment these lines after testing
# N_FOLDS = 5
N_SEEDS = 3
ENSEMBLE_SEEDS = [42, 123, 456]

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


def to_01(arr, axis_idx, scaler):
    """Forward-normalise a (N, 101) array along one axis."""
    return (arr - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]


# ------------------------------
# Dataset utilities with Gaussian noise (unchanged)
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
            self.channel_noise_stds = torch.zeros(42)
            
            for ch in range(42):
                channel_data = self.X[:, ch, :].flatten()
                channel_std = channel_data.std().item()
                self.channel_noise_stds[ch] = noise_std_fraction * channel_std
            
            print(f"  Noise augmentation enabled with std fraction={noise_std_fraction}")
            print(f"  Channel noise STDs range: [{self.channel_noise_stds.min():.4f}, {self.channel_noise_stds.max():.4f}]")
        
        # Load participant IDs if available
        pid_key = f"participant_{sess}"
        if pid_key in data:
            self.participants = data[pid_key]
        elif f"PID_{sess}" in data:
            self.participants = data[f"PID_{sess}"]
        else:
            self.participants = np.arange(len(X))
        
        # Initialize training mode to avoid AttributeError on first access
        self.training = False
        
        print(f"Loaded {len(self.X)} samples for {sess} set, fold {fold}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # Shape: (42, 101)
        y = self.Y[idx]
        
        # Apply Gaussian noise during training only
        if self.apply_noise and self.training:
            noise = torch.randn_like(x)
            
            for ch in range(42):
                noise[ch, :] *= self.channel_noise_stds[ch]
            
            x = x + noise
        
        return x, y
    
    def set_training(self, mode):
        """Set training mode for noise application."""
        self.training = mode


def get_train_val_indices(participants, val_ratio=0.2, random_state=42):
    """Split indices into train/validation subsets grouped by participant."""
    df = pd.DataFrame({'pid': participants})
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['pid']))
    return train_idx, val_idx


def get_base_dataset(loader_dataset):
    """Get underlying CNNDataset from a potentially wrapped dataset."""
    if isinstance(loader_dataset, torch.utils.data.Subset):
        return loader_dataset.dataset
    return loader_dataset


# ------------------------------
# Model definition (unchanged)
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


class MultiScaleCNN(nn.Module):
    """Configurable multi-branch 1D CNN with proper BatchNorm."""

    def __init__(self, kernels, channels, n_layers, pooling,
                 dropout_conv, dropout_fc, num_features=42, num_outputs=303):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            layers = []
            in_ch = num_features
            for i in range(n_layers):
                out_ch = channels[min(i, len(channels) - 1)]
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k,
                                         padding=k // 2))
                layers.append(SafeBatchNorm1d(out_ch))
                layers.append(nn.ReLU())
                if dropout_conv > 0:
                    layers.append(nn.Dropout(dropout_conv))
                in_ch = out_ch
            if pooling == 'adaptive_avg':
                layers.append(nn.AdaptiveAvgPool1d(1))
            elif pooling == 'adaptive_max':
                layers.append(nn.AdaptiveMaxPool1d(1))
            else:  # global_avg over entire length
                layers.append(nn.AvgPool1d(kernel_size=101))
            self.branches.append(nn.Sequential(*layers))
        fc_in = channels[min(n_layers - 1, len(channels) - 1)] * len(kernels)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in),
            nn.BatchNorm1d(fc_in),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_in, num_outputs),
        )

    def forward(self, x):  # x: (batch, 42, 101)
        outs = [branch(x).squeeze(-1) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        return self.fc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))


# ------------------------------
# VARIANCE ENCOURAGEMENT LOSS
# ------------------------------
class VarianceEncouragedLoss(nn.Module):
    """
    Combined loss with variance encouragement penalty.
    
    Computes base loss + variance penalty when prediction spread < threshold * target spread.
    Uses inverse-transformed predictions to match evaluation metrics.
    """
    
    def __init__(self, base_criterion, scaler, 
                 variance_coeff=0.03, variance_threshold=0.90, 
                 min_batch_size=4):
        super().__init__()
        self.base_criterion = base_criterion
        self.scaler = scaler
        self.variance_coeff = variance_coeff
        self.variance_threshold = variance_threshold
        self.min_batch_size = min_batch_size
        
        # Precompute scaler arrays for efficiency
        self.scale_tensor = torch.tensor([scaler.scale_[0], scaler.scale_[1], scaler.scale_[2]], 
                                        dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.min_tensor = torch.tensor([scaler.min_[0], scaler.min_[1], scaler.min_[2]], 
                                      dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Logging
        self.variance_penalty_history = []
        self.base_loss_history = []
    
    def inverse_transform_batch(self, y_normalized, device):
        """Apply inverse transform to get back to original units (degrees)."""
        batch_size = y_normalized.size(0)
        
        # Reshape to (batch, 3, 101) for axis-wise processing
        y_reshaped = y_normalized.view(batch_size, 3, 101)
        
        # Apply inverse transform per axis
        scale = self.scale_tensor.to(device).view(1, 3, 1)  # (1, 3, 1)
        min_val = self.min_tensor.to(device).view(1, 3, 1)   # (1, 3, 1)
        
        y_transformed = y_reshaped * scale + min_val
        
        return y_transformed  # (batch, 3, 101)
    
    def compute_variance_penalty(self, pred_transformed, target_transformed):
        """
        Compute variance encouragement penalty.
        
        Args:
            pred_transformed: (batch, 3, 101) in original units
            target_transformed: (batch, 3, 101) in original units
        
        Returns:
            variance_penalty: scalar tensor
        """
        batch_size = pred_transformed.size(0)
        if batch_size < self.min_batch_size:
            return torch.tensor(0.0, device=pred_transformed.device)
        
        penalty = torch.tensor(0.0, device=pred_transformed.device)
        n_axes_penalized = 0
        
        # Compute penalty per axis
        for axis in range(3):  # X, Y, Z
            pred_axis = pred_transformed[:, axis, :]  # (batch, 101)
            target_axis = target_transformed[:, axis, :]  # (batch, 101)
            
            # Compute across-trial standard deviation for each time point
            pred_std = torch.std(pred_axis, dim=0, unbiased=True)    # (101,)
            target_std = torch.std(target_axis, dim=0, unbiased=True)  # (101,)
            
            # Mean standard deviation across time
            pred_mean_std = torch.mean(pred_std)
            target_mean_std = torch.mean(target_std)
            
            # Check if penalty should activate
            if target_mean_std > 1e-6:  # Avoid division by zero
                ratio = pred_mean_std / target_mean_std
                
                if ratio < self.variance_threshold:
                    # Penalty: squared difference from threshold
                    axis_penalty = (self.variance_threshold - ratio) ** 2
                    penalty += axis_penalty
                    n_axes_penalized += 1
        
        # Average penalty across penalized axes
        if n_axes_penalized > 0:
            penalty = penalty / n_axes_penalized
        
        return penalty
    
    def forward(self, y_pred, y_true):
        """
        Compute combined loss: base_loss + variance_coeff * variance_penalty
        
        Args:
            y_pred: (batch, 303) normalized predictions
            y_true: (batch, 303) normalized targets
        
        Returns:
            total_loss: scalar tensor
        """
        device = y_pred.device
        
        # Base loss in normalized space
        base_loss = self.base_criterion(y_pred, y_true)
        
        # Variance penalty in transformed space
        variance_penalty = torch.tensor(0.0, device=device)
        
        if self.variance_coeff > 0 and y_pred.size(0) >= self.min_batch_size:
            # Transform to original units
            pred_transformed = self.inverse_transform_batch(y_pred, device)
            target_transformed = self.inverse_transform_batch(y_true, device)
            
            # Compute variance penalty
            variance_penalty = self.compute_variance_penalty(pred_transformed, target_transformed)
        
        # Combined loss
        total_loss = base_loss + self.variance_coeff * variance_penalty
        
        # Log for analysis
        self.base_loss_history.append(base_loss.item())
        self.variance_penalty_history.append(variance_penalty.item())
        
        return total_loss, base_loss.item(), variance_penalty.item()


# ------------------------------
# Metrics (unchanged)
# ------------------------------
def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler):
    """nRMSE for one axis, averaged over the current mini-batch."""
    idx = {'x': 0, 'y': 1, 'z': 2}[axis]

    pred_axis = pred.view(-1, 3, 101)[:, idx, :]
    targ_axis = target.view(-1, 3, 101)[:, idx, :]

    rmse = torch.sqrt(torch.mean((pred_axis - targ_axis) ** 2, dim=1))
    nrmse = 100 * torch.mean(rmse)  # range = 1.0 in normalised space
    return nrmse.item()


def ensemble_evaluation(models, loader, device, scaler, set_name="TEST",
                       training_ranges=None, epoch=None, print_results=True,
                       return_predictions=False):
    """Comprehensive evaluation for ensemble of models (unchanged)."""
    for model in models:
        model.eval()
    
    if training_ranges is None:
        training_ranges = {
            'X': 1.0,
            'Y': 1.0,
            'Z': 1.0
        }
    
    all_ensemble_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            batch_preds = []
            for model in models:
                pred = model(X)
                batch_preds.append(pred.cpu().numpy())
            
            ensemble_pred = np.mean(batch_preds, axis=0)
            
            all_ensemble_preds.append(ensemble_pred)
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_ensemble_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    if return_predictions:
        return all_preds, all_targets
    
    # [Rest of evaluation code remains the same as in original script]
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
            print(f"{set_name} SET EVALUATION (ENSEMBLE + VARIANCE LOSS) - EPOCH {epoch}")
        else:
            print(f"\n{'='*80}")
            print(f"{set_name} SET EVALUATION (ENSEMBLE + VARIANCE LOSS) (Final)")
        print(f"{'='*80}")
        print(f"Ensemble size: {len(models)} models")
        if GAUSSIAN_NOISE_ENABLED and "TRAIN" in set_name.upper():
            print(f"Gaussian noise augmentation: ENABLED (std fraction={NOISE_STD_FRACTION})")
        if VARIANCE_LOSS_ENABLED and "TRAIN" in set_name.upper():
            print(f"Variance encouragement loss: ENABLED (coeff={VARIANCE_LOSS_COEFFICIENT}, threshold={VARIANCE_THRESHOLD})")
        
        print("\nTraining data ranges (normalization denominators):")
        print("-"*70)
        for axis in ['X', 'Y', 'Z']:
            range_val = axis_data[axis]['range']
            print(f"{axis}: range={range_val:.2f}°")
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
            nrmse = 100 * rmse  # range = 1.0 in normalised space
            
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
        global_nrmse = 100 * global_rmse  # range = 1.0 in normalised space
        
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
            print(f"  Training range: {train_range:.2f}°")
            print(f"  Per-trial Corr: {per_trial_corrs.mean():.3f}±{per_trial_corrs.std():.3f}")
            print(f"  Global Corr: {global_corr:.3f}")
            print(f"  Global RMSE: {global_rmse:.3f} (normalised space)")
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
        print(f"{set_name} SUMMARY (ENSEMBLE + VARIANCE LOSS) - " + (f"EPOCH {epoch}" if epoch is not None else "FINAL"))
        print(f"{'='*80}")
        print(f"Average global correlation: {avg_global_corr:.3f}")
        print(f"Average global RMSE: {avg_global_rmse:.3f} (normalised space)")
        print(f"Average global nRMSE: {avg_global_nrmse:.2f}%")
        print(f"Average SD ratio: {avg_sd_ratio:.3f}")
        
        if set_name == "TRAIN" or "TRAINING" in set_name:
            if avg_sd_ratio > 0.9:
                print("✅ SD ratio > 0.9: Ensemble captures excellent variance")
            elif avg_sd_ratio > 0.8:
                print("⚠️  SD ratio 0.8-0.9: Good variance capture")
            else:
                print("🔧 SD ratio < 0.8: Variance loss should help improve this")
    
    results['summary'] = {
        'avg_global_corr': avg_global_corr,
        'avg_global_rmse': avg_global_rmse, 
        'avg_global_nrmse': avg_global_nrmse,
        'avg_sd_ratio': avg_sd_ratio
    }
    
    return results


# ------------------------------
# Training utilities with variance-encouraged loss
# ------------------------------
def create_optimizer(params, opt_name, lr, weight_decay):
    if opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == 'nadam':
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)


def create_base_loss(name):
    if name == 'mse':
        return nn.MSELoss()
    if name == 'mae':
        return nn.L1Loss()
    if name == 'rmse':
        return RMSELoss()
    return nn.HuberLoss()


def train_epoch_with_variance_loss(model, loader, optimizer, criterion, device, scaler, grad_clip):
    """Train one epoch with variance-encouraged loss."""
    model.train()
    
    # Enable training mode in dataset to apply noise
    base_dataset = get_base_dataset(loader.dataset)
    if hasattr(base_dataset, 'set_training'):
        base_dataset.set_training(True)
    
    total_loss = 0
    total_base_loss = 0
    total_variance_penalty = 0
    x_err = y_err = z_err = 0
    n_batches = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        
        # Get combined loss, base loss, and variance penalty
        if isinstance(criterion, VarianceEncouragedLoss):
            loss, base_loss_val, variance_penalty_val = criterion(out, y)
            total_base_loss += base_loss_val * X.size(0)
            total_variance_penalty += variance_penalty_val * X.size(0)
        else:
            loss = criterion(out, y)
            total_base_loss += loss.item() * X.size(0)
            total_variance_penalty += 0.0
        
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        x_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'x', scaler)
        y_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'y', scaler)
        z_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'z', scaler)
        n_batches += 1
    
    # Disable training mode after epoch
    if hasattr(base_dataset, 'set_training'):
        base_dataset.set_training(False)
    
    n = len(loader.dataset)
    return (total_loss / n, total_base_loss / n, total_variance_penalty / n, 
            x_err / n_batches, y_err / n_batches, z_err / n_batches)


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


def train_model_with_coeff(fold, seed, train_loader, val_loader, scaler, coeff):
    """Train a single model for coefficient search."""
    set_seed(seed)

    model = MultiScaleCNN(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        pooling=BEST_PARAMS['pooling'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc']
    ).to(device)

    base_criterion = create_base_loss(BEST_PARAMS['loss'])

    if coeff == 0.0:
        criterion = base_criterion
    else:
        criterion = VarianceEncouragedLoss(
            base_criterion=base_criterion,
            scaler=scaler,
            variance_coeff=coeff,
            variance_threshold=VARIANCE_THRESHOLD,
            min_batch_size=MIN_BATCH_SIZE_FOR_VARIANCE,
        )

    optimizer = create_optimizer(
        model.parameters(),
        BEST_PARAMS['optimizer'],
        BEST_PARAMS['lr'],
        BEST_PARAMS['weight_decay']
    )

    for _ in range(SEARCH_EPOCHS):
        train_epoch_with_variance_loss(
            model, train_loader, optimizer, criterion,
            device, scaler, BEST_PARAMS['grad_clip']
        )

    train_metrics = ensemble_evaluation(
        [model], train_loader, device, scaler,
        set_name="TRAIN", print_results=False
    )
    val_metrics = ensemble_evaluation(
        [model], val_loader, device, scaler,
        set_name="VAL", print_results=False
    )

    result = {
        'val_sd_ratio': val_metrics['summary']['avg_sd_ratio'],
        'val_nrmse': val_metrics['summary']['avg_global_nrmse'],
        'train_sd_ratio': train_metrics['summary']['avg_sd_ratio'],
        'train_nrmse': train_metrics['summary']['avg_global_nrmse'],
        'train_val_gap': val_metrics['summary']['avg_global_nrmse'] - train_metrics['summary']['avg_global_nrmse'],
    }

    return model, result


def train_single_model_with_variance_loss(fold, seed_idx, seed, train_loader, test_loader,
                                        scaler, log_dir_base):
    """Train a single model with variance-encouraged loss."""
    print(f"\n  Training ensemble member {seed_idx+1}/{N_SEEDS} (seed={seed}) WITH VARIANCE LOSS")
    
    # Set seed for this model
    set_seed(seed)
    
    # Initialize model
    model = MultiScaleCNN(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        pooling=BEST_PARAMS['pooling'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc']
    ).to(device)
    
    # Training setup with variance-encouraged loss
    base_criterion = create_base_loss(BEST_PARAMS['loss'])
    
    if VARIANCE_LOSS_ENABLED:
        criterion = VarianceEncouragedLoss(
            base_criterion=base_criterion,
            scaler=scaler,
            variance_coeff=VARIANCE_LOSS_COEFFICIENT,
            variance_threshold=VARIANCE_THRESHOLD,
            min_batch_size=MIN_BATCH_SIZE_FOR_VARIANCE
        )
        print(f"    Variance loss enabled: coeff={VARIANCE_LOSS_COEFFICIENT}, threshold={VARIANCE_THRESHOLD}")
    else:
        criterion = base_criterion
        print(f"    Using standard loss: {BEST_PARAMS['loss']}")
    
    optimizer = create_optimizer(
        model.parameters(), 
        BEST_PARAMS['optimizer'],
        BEST_PARAMS['lr'], 
        BEST_PARAMS['weight_decay']
    )
    
    # Setup logging for this model
    log_dir = join(log_dir_base, f'seed_{seed}')
    os.makedirs(log_dir, exist_ok=True)
    writer_train = SummaryWriter(join(log_dir, 'train'))
    writer_test = SummaryWriter(join(log_dir, 'test'))

    # Training loop with variance loss tracking
    for epoch in range(ENSEMBLE_EPOCHS):
        # Training step (with noise and variance loss)
        if VARIANCE_LOSS_ENABLED:
            tr_loss, tr_base_loss, tr_var_penalty, tr_x, tr_y, tr_z = train_epoch_with_variance_loss(
                model, train_loader, optimizer, criterion,
                device, scaler, BEST_PARAMS['grad_clip']
            )
            
            # Log detailed variance loss metrics
            writer_train.add_scalar('loss_total', tr_loss, epoch)
            writer_train.add_scalar('loss_base', tr_base_loss, epoch)
            writer_train.add_scalar('loss_variance_penalty', tr_var_penalty, epoch)
            writer_train.add_scalar('variance_penalty_ratio', tr_var_penalty / tr_base_loss if tr_base_loss > 0 else 0, epoch)
        else:
            tr_loss, _, _, tr_x, tr_y, tr_z = train_epoch_with_variance_loss(
                model, train_loader, optimizer, criterion,
                device, scaler, BEST_PARAMS['grad_clip']
            )
            writer_train.add_scalar('loss', tr_loss, epoch)
        
        writer_train.add_scalar('nrmse_mean', (tr_x + tr_y + tr_z) / 3, epoch)
        
        # Progress update every 10 epochs
        if (epoch + 1) % 10 == 0:
            if VARIANCE_LOSS_ENABLED:
                print(f"    Epoch {epoch+1:3d}: Total={tr_loss:.4f}, Base={tr_base_loss:.4f}, "
                      f"VarPenalty={tr_var_penalty:.4f}, nRMSE={(tr_x + tr_y + tr_z) / 3:.2f}%")
            else:
                print(f"    Epoch {epoch+1:3d}: Loss={tr_loss:.4f}, "
                      f"nRMSE mean={(tr_x + tr_y + tr_z) / 3:.2f}%")
    
    # Close writers
    writer_train.close()
    writer_test.close()
    
    return model


def multi_seed_ensemble_training_with_variance_loss():
    """Multi-seed ensemble training with variance encouragement loss."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    check_data_availability()
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(ENSEMBLE_PREDS_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 3b: MULTI-SEED ENSEMBLE TRAINING WITH VARIANCE ENCOURAGEMENT LOSS")
    print("🧪 TEST MODE: Running with 1 fold, 1 seed for pipeline verification")
    print("="*80)
    print("📐  NOTE: All metrics / files are now in 0-1 normalised space (matching makeEstimationWithPDF notebooks)")
    print(f"Training {N_SEEDS} model(s) per fold with seeds: {ENSEMBLE_SEEDS}")
    print(f"Training for {ENSEMBLE_EPOCHS} epochs per model")
    print(f"Using complete train+val data (no validation split)")
    print(f"Gaussian noise augmentation: {'ENABLED' if GAUSSIAN_NOISE_ENABLED else 'DISABLED'}")
    if GAUSSIAN_NOISE_ENABLED:
        print(f"  Noise STD fraction: {NOISE_STD_FRACTION} (of channel range)")
    print(f"Variance encouragement loss: {'ENABLED' if VARIANCE_LOSS_ENABLED else 'DISABLED'}")
    if VARIANCE_LOSS_ENABLED:
        print(f"  Variance coefficient: {VARIANCE_LOSS_COEFFICIENT} ({VARIANCE_LOSS_COEFFICIENT*100:.1f}% of base loss)")
        print(f"  Variance threshold: {VARIANCE_THRESHOLD} (penalty when pred_spread < {VARIANCE_THRESHOLD*100:.0f}% target_spread)")
        print(f"  Min batch size for variance loss: {MIN_BATCH_SIZE_FOR_VARIANCE}")
    print("\n🎯 TESTING GOALS:")
    print("   1. Verify variance loss improves SD ratio (target: >0.8)")
    print("   2. Ensure nRMSE doesn't inflate significantly")
    print("   3. Validate penalty activates appropriately during training")
    print(f"   4. If successful, switch to full 5-fold 3-seed training")
    print(f"Optimal hyperparameters: {BEST_PARAMS}")
    print("="*80)
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'step': 'multi_seed_ensemble_variance_loss',
        'n_seeds': N_SEEDS,
        'seeds': ENSEMBLE_SEEDS,
        'epochs': ENSEMBLE_EPOCHS,
        'gaussian_noise': {
            'enabled': GAUSSIAN_NOISE_ENABLED,
            'std_fraction': NOISE_STD_FRACTION,
            'description': 'Gaussian noise added to inputs during training only'
        },
        'variance_loss': {
            'enabled': VARIANCE_LOSS_ENABLED,
            'coefficient': VARIANCE_LOSS_COEFFICIENT,
            'threshold': VARIANCE_THRESHOLD,
            'min_batch_size': MIN_BATCH_SIZE_FOR_VARIANCE,
            'description': 'Penalty when prediction spread < threshold * target spread'
        },
        'best_params': BEST_PARAMS,
        'data_split': 'full_train_no_validation',
        'note': 'Step 3b: variance encouragement to improve SD-ratio for limited sample size'
    }
    
    with open(join(RESULTS_DIR, f'ensemble_variance_config_{timestamp}.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Store results for all folds
    all_fold_results = {}

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold} - ENSEMBLE WITH VARIANCE ENCOURAGEMENT")
        print(f"{'='*60}")
        
        # Load scaler and compute per-fold training ranges
        scaler_path = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
        scaler = load(open(scaler_path, 'rb'))
        
        # Compute training ranges for this fold (same as notebook)
        training_ranges = {
             'X': scaler.data_max_[0] - scaler.data_min_[0],
             'Y': scaler.data_max_[1] - scaler.data_min_[1],
             'Z': scaler.data_max_[2] - scaler.data_min_[2],
             
        }

        
        print(f"Training ranges for fold {fold}: X={training_ranges['X']:.2f}°, "
              f"Y={training_ranges['Y']:.2f}°, Z={training_ranges['Z']:.2f}°")
        
        # Load FULL training data with noise augmentation enabled
        full_train_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'train', fold,
            apply_noise=GAUSSIAN_NOISE_ENABLED,
            noise_std_fraction=NOISE_STD_FRACTION,
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
        
        # Create data loaders - CRITICAL: test loader must have drop_last=False
        train_loader = DataLoader(full_train_dataset, 
                                  batch_size=BEST_PARAMS['batch_size'], 
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=BEST_PARAMS['batch_size'], 
                                 shuffle=False, drop_last=False)  # Keep all test samples

        # Train ensemble members with variance loss
        ensemble_models = []
        log_dir_base = join(LOGS_DIR, f'fold_{fold}')
        
        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            model = train_single_model_with_variance_loss(
                fold, seed_idx, seed, train_loader, test_loader,
                scaler, log_dir_base
            )
            ensemble_models.append(model)
            
            # Save individual model
            model_path = join(MODELS_DIR, f'fold_{fold}_seed_{seed}_variance_model.pt')
            torch.save(model.state_dict(), model_path)

        # Ensemble evaluation
        print(f"\n{'='*60}")
        print(f"ENSEMBLE EVALUATION WITH VARIANCE LOSS - FOLD {fold}")
        print(f"{'='*60}")
        
        # Disable noise for evaluation
        full_train_dataset.set_training(False)
        
        # Training set evaluation (without noise)
        train_results = ensemble_evaluation(
            ensemble_models, train_loader, device, scaler,
            set_name="ENSEMBLE TRAINING", print_results=True
        )
        
        # Test set evaluation
        test_results = ensemble_evaluation(
            ensemble_models, test_loader, device, scaler,
            set_name="ENSEMBLE TEST", print_results=True
        )
        
        # Print variance loss effectiveness summary
        print(f"\n{'='*60}")
        print(f"VARIANCE LOSS EFFECTIVENESS - FOLD {fold}")
        print(f"{'='*60}")
        train_sd_ratio = train_results['summary']['avg_sd_ratio']
        test_sd_ratio = test_results['summary']['avg_sd_ratio']
        
        print(f"Training SD ratio: {train_sd_ratio:.3f}")
        print(f"Test SD ratio: {test_sd_ratio:.3f}")
        
        if train_sd_ratio > 0.85:
            print("✅ Excellent SD ratio improvement with variance loss!")
        elif train_sd_ratio > 0.80:
            print("⚠️  Good SD ratio - variance loss helping")
        else:
            print("🔧 SD ratio still low - consider increasing variance coefficient")
            print(f"   Current coefficient: {VARIANCE_LOSS_COEFFICIENT}")
            print(f"   Suggested: try {VARIANCE_LOSS_COEFFICIENT * 1.5:.3f} to {VARIANCE_LOSS_COEFFICIENT * 2:.3f}")
        
        # Save ensemble predictions
        print("\nSaving ensemble predictions...")
        train_preds, train_targets = ensemble_evaluation(
            ensemble_models, train_loader, device, scaler,
            return_predictions=True
        )
        test_preds, test_targets = ensemble_evaluation(
            ensemble_models, test_loader, device, scaler,
            return_predictions=True
        )
        
        pred_file = join(ENSEMBLE_PREDS_DIR, f'fold_{fold}_ensemble_variance_predictions.npz')
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
            'gaussian_noise': {
                'enabled': GAUSSIAN_NOISE_ENABLED,
                'std_fraction': NOISE_STD_FRACTION
            },
            'variance_loss': {
                'enabled': VARIANCE_LOSS_ENABLED,
                'coefficient': VARIANCE_LOSS_COEFFICIENT,
                'threshold': VARIANCE_THRESHOLD,
                'min_batch_size': MIN_BATCH_SIZE_FOR_VARIANCE
            },
            'training': train_results,
            'test': test_results,
            'hyperparameters': BEST_PARAMS
        }
        
        all_fold_results[f'fold_{fold}'] = fold_results
        
        # Save individual fold results
        results_file = join(RESULTS_DIR, f'fold_{fold}_ensemble_variance_results.json')
        with open(results_file, 'w') as f:
            json.dump(convert_numpy(fold_results), f, indent=2)
        
        # Save entire ensemble (all models) for potential future use
        ensemble_path = join(MODELS_DIR, f'fold_{fold}_ensemble_variance_models.pt')
        torch.save({
            'models': [model.state_dict() for model in ensemble_models],
            'config': BEST_PARAMS,
            'seeds': ENSEMBLE_SEEDS,
            'gaussian_noise': {
                'enabled': GAUSSIAN_NOISE_ENABLED,
                'std_fraction': NOISE_STD_FRACTION
            },
            'variance_loss': {
                'enabled': VARIANCE_LOSS_ENABLED,
                'coefficient': VARIANCE_LOSS_COEFFICIENT,
                'threshold': VARIANCE_THRESHOLD
            }
        }, ensemble_path)

    # Summary across all folds
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING WITH VARIANCE LOSS COMPLETED - SUMMARY ACROSS ALL FOLDS")
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
    sd_gap = avg_train_sd - avg_test_sd
    
    print(f"\nGeneralization gaps:")
    print(f"  Correlation gap: {corr_gap:.3f} (train - test)")
    print(f"  nRMSE gap: {nrmse_gap:.2f}% (test - train)")
    print(f"  SD ratio gap: {sd_gap:.3f} (train - test)")
    
    print(f"\n🎯 VARIANCE LOSS EFFECTIVENESS ASSESSMENT:")
    print(f"{'='*50}")
    if avg_train_sd > 0.85 and avg_test_sd > 0.80:
        print("✅ EXCELLENT: Variance loss successfully improved SD ratios!")
        print("   Both training and test SD ratios are in excellent range")
    elif avg_train_sd > 0.80:
        print("⚠️  GOOD: Variance loss helped improve SD ratios")
        print("   Training SD ratio improved, test may need more data")
    else:
        print("🔧 MODERATE: Some improvement, but SD ratio still low")
        print("   Consider tuning variance loss parameters:")
        print(f"   - Increase coefficient from {VARIANCE_LOSS_COEFFICIENT} to {VARIANCE_LOSS_COEFFICIENT * 1.5:.3f}")
        print(f"   - Or adjust threshold from {VARIANCE_THRESHOLD} to 0.95")
    
    if corr_gap < 0.05 and nrmse_gap < 2.0:
        print("✅ Excellent generalization: minimal train-test gap")
    elif corr_gap < 0.10 and nrmse_gap < 5.0:
        print("⚠️  Good generalization: moderate train-test gap")
    else:
        print("❌ Poor generalization: large train-test gap")
    
    # Save overall summary
    summary = {
        'timestamp': timestamp,
        'n_folds': N_FOLDS,
        'n_seeds_per_fold': N_SEEDS,
        'seeds': ENSEMBLE_SEEDS,
        'epochs': ENSEMBLE_EPOCHS,
        'gaussian_noise': {
            'enabled': GAUSSIAN_NOISE_ENABLED,
            'std_fraction': NOISE_STD_FRACTION,
            'benefit': 'Reduces overfitting by making model robust to small input variations'
        },
        'variance_loss': {
            'enabled': VARIANCE_LOSS_ENABLED,
            'coefficient': VARIANCE_LOSS_COEFFICIENT,
            'threshold': VARIANCE_THRESHOLD,
            'min_batch_size': MIN_BATCH_SIZE_FOR_VARIANCE,
            'benefit': 'Encourages predictions to maintain realistic variance, improving SD ratio'
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
                'nrmse_gap': nrmse_gap,
                'sd_ratio_gap': sd_gap
            }
        },
        'variance_loss_effectiveness': {
            'sd_ratio_improvement': 'See individual fold results',
            'recommendation': 'Compare with baseline ensemble without variance loss'
        },
        'fold_results': all_fold_results
    }
    
    summary_file = join(RESULTS_DIR, f'ensemble_variance_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2)
    
    print(f"\n{'='*80}")
    print("VARIANCE ENCOURAGEMENT TEST COMPLETED!")
    print(f"{'='*80}")
    print(f"🧪 TEST RESULTS FOR FOLD 0:")
    print(f"   Training SD ratio: {avg_train_sd:.3f}")
    print(f"   Test SD ratio: {avg_test_sd:.3f}")
    print(f"   Training nRMSE: {avg_train_nrmse:.2f}%")
    print(f"   Test nRMSE: {avg_test_nrmse:.2f}%")
    print()
    
    if avg_train_sd > 0.8:
        print("✅ SUCCESS: Variance loss improved SD ratio! Ready for full training.")
        print("📝 NEXT STEPS:")
        print("   1. Uncomment full training parameters in script:")
        print("      N_FOLDS = 5")
        print("      N_SEEDS = 3") 
        print("      ENSEMBLE_SEEDS = [42, 123, 456]")
        print("   2. Re-run script for complete 5-fold 3-seed ensemble")
    elif avg_train_sd > 0.75:
        print("⚠️  PARTIAL SUCCESS: Some improvement, consider tuning:")
        print(f"   - Increase VARIANCE_LOSS_COEFFICIENT to {VARIANCE_LOSS_COEFFICIENT * 1.5:.3f}")
        print("   - Or adjust VARIANCE_THRESHOLD to 0.95")
        print("   - Test again before full training")
    else:
        print("🔧 NEEDS TUNING: SD ratio still low, try:")
        print(f"   - Increase VARIANCE_LOSS_COEFFICIENT to {VARIANCE_LOSS_COEFFICIENT * 2:.3f}")
        print("   - Or decrease VARIANCE_THRESHOLD to 0.85")
        print("   - Test with higher penalty before full training")
    
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")
    print(f"Predictions saved to: {ENSEMBLE_PREDS_DIR}")
    print("\nKey improvements from variance encouragement loss:")
    print("1. Prevents under-dispersed predictions (low SD ratios)")
    print("2. Encourages realistic prediction variance without sacrificing RMSE")
    print("3. Particularly effective for limited sample size scenarios")
    print("4. Penalty only activates when prediction spread falls below target spread")
    print(f"5. Current settings: {VARIANCE_LOSS_COEFFICIENT*100:.1f}% penalty when pred_spread < {VARIANCE_THRESHOLD*100:.0f}% target_spread")


def search_optimal_variance_coefficient():
    """Search for the optimal variance loss coefficient using validation data."""

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(VARIANCE_SEARCH_RESULTS_DIR, exist_ok=True)
    os.makedirs(VARIANCE_SEARCH_MODELS_DIR, exist_ok=True)
    os.makedirs(VARIANCE_SEARCH_LOGS_DIR, exist_ok=True)
    os.makedirs(VARIANCE_SEARCH_DATA_DIR, exist_ok=True)

    all_results = {}

    for fold in range(N_FOLDS):
        scaler_path = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
        scaler = load(open(scaler_path, 'rb'))

        full_train_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'train', fold,
            apply_noise=GAUSSIAN_NOISE_ENABLED,
            noise_std_fraction=NOISE_STD_FRACTION,
            scaler=scaler
        )

        train_idx, val_idx = get_train_val_indices(full_train_dataset.participants)
        train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_train_dataset, val_idx)

        train_loader = DataLoader(train_subset,
                                  batch_size=BEST_PARAMS['batch_size'],
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset,
                                batch_size=BEST_PARAMS['batch_size'],
                                shuffle=False, drop_last=False)

        fold_results = {}

        for coeff in VARIANCE_COEFFS_TO_TEST:
            seed_metrics = []
            for seed in SEARCH_SEEDS:
                model, metrics = train_model_with_coeff(fold, seed, train_loader, val_loader, scaler, coeff)
                seed_metrics.append(metrics)

                model_path = join(VARIANCE_SEARCH_MODELS_DIR, f'fold_{fold}_seed_{seed}_coeff_{coeff}.pt')
                torch.save(model.state_dict(), model_path)

            avg_val_sd = np.mean([m['val_sd_ratio'] for m in seed_metrics])
            avg_val_nrmse = np.mean([m['val_nrmse'] for m in seed_metrics])
            avg_train_gap = np.mean([m['train_val_gap'] for m in seed_metrics])

            fold_results[f'coeff_{coeff}'] = {
                'coefficient': coeff,
                'avg_val_sd_ratio': float(avg_val_sd),
                'avg_val_nrmse': float(avg_val_nrmse),
                'avg_train_val_gap': float(avg_train_gap),
                'seed_results': seed_metrics
            }

        fold_file = join(VARIANCE_SEARCH_DATA_DIR, f'fold_{fold}_variance_coefficient_search.json')
        with open(fold_file, 'w') as f:
            json.dump(convert_numpy(fold_results), f, indent=2)

        all_results[f'fold_{fold}'] = fold_results

    # Determine best coefficient across folds using mean validation SD ratio
    coeff_scores = {}
    for fold_data in all_results.values():
        for key, val in fold_data.items():
            coeff = val['coefficient']
            coeff_scores.setdefault(coeff, []).append(val['avg_val_sd_ratio'])

    best_coeff = max(coeff_scores.items(), key=lambda x: np.mean(x[1]))[0]

    baseline = all_results['fold_0']['coeff_0.0']
    best_fold0 = all_results['fold_0'][f'coeff_{best_coeff}']

    improvement_summary = {
        'sd_ratio_improvement': best_fold0['avg_val_sd_ratio'] - baseline['avg_val_sd_ratio'],
        'nrmse_cost': best_fold0['avg_val_nrmse'] - baseline['avg_val_nrmse']
    }

    summary = {
        'recommended_coefficient': best_coeff,
        'baseline_performance': {
            'coeff_0.0': {
                'avg_val_sd_ratio': baseline['avg_val_sd_ratio'],
                'avg_val_nrmse': baseline['avg_val_nrmse']
            }
        },
        'best_performance': {
            f'coeff_{best_coeff}': {
                'avg_val_sd_ratio': best_fold0['avg_val_sd_ratio'],
                'avg_val_nrmse': best_fold0['avg_val_nrmse']
            }
        },
        'improvement_summary': improvement_summary
    }

    summary_file = join(VARIANCE_SEARCH_RESULTS_DIR, f'variance_coefficient_search_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2)

    optimal_config = {
        'VARIANCE_LOSS_COEFFICIENT': best_coeff,
        'VARIANCE_THRESHOLD': VARIANCE_THRESHOLD,
        'MIN_BATCH_SIZE_FOR_VARIANCE': MIN_BATCH_SIZE_FOR_VARIANCE,
        'validation_performance': {
            'avg_val_sd_ratio': best_fold0['avg_val_sd_ratio'],
            'avg_val_nrmse': best_fold0['avg_val_nrmse']
        },
        'usage_note': 'Replace VARIANCE_LOSS_COEFFICIENT in original script with this value'
    }

    config_file = join(VARIANCE_SEARCH_RESULTS_DIR, 'optimal_variance_config.json')
    with open(config_file, 'w') as f:
        json.dump(convert_numpy(optimal_config), f, indent=2)

    print("\n🎯 READY TO USE:")
    print(f"   Optimal coefficient: {best_coeff}")
    print(f"   Config saved to: {config_file}")
    print(f"   Replace VARIANCE_LOSS_COEFFICIENT = {best_coeff} in original script")


def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


if __name__ == '__main__':
    search_optimal_variance_coefficient()