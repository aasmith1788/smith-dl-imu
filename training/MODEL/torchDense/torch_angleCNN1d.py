# Step 2: Full retrain using optimal hyperparameters on complete train+val data
# Multi-scale 1D CNN full retrain script for IMU joint angle prediction

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


# Set seed at module level
set_seed(42)


# ------------------------------
# Configuration - OPTIMAL PARAMETERS FROM OPTUNA
# ------------------------------
DATASET_NAME = 'IWALQQ_1st_correction'
DATA_TYPE = 'angle'

BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)

# Output directories for full retrain
RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\MultiScaleCNN_full_retrain'
MODELS_DIR = join(RESULTS_DIR, 'models')
LOGS_DIR = join(RESULTS_DIR, 'logs')

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

# Training parameters for full retrain
N_FOLDS = 5
FULL_RETRAIN_EPOCHS = 15 #topping, train for fixed epochs
MAX_PARAMETERS = 500_000

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Dataset directory: {DATASET_DIR}")


# ------------------------------
# Dataset utilities
# ------------------------------
class CNNDataset(Dataset):
    """Dataset that reshapes flattened IMU sequences to (42, 101)."""

    def __init__(self, data_dir, data_type, sess, fold):
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
        
        # Load participant IDs if available
        pid_key = f"participant_{sess}"
        if pid_key in data:
            self.participants = data[pid_key]
        elif f"PID_{sess}" in data:
            self.participants = data[f"PID_{sess}"]
        else:
            self.participants = np.arange(len(X))
        
        print(f"Loaded {len(self.X)} samples for {sess} set, fold {fold}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ------------------------------
# Model definition with BatchNorm safety
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
# Metrics - Fixed to match your evaluation pipeline
# ------------------------------
def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler):
    """Calculate nRMSE for a specific axis using the same double-normalization as evaluation."""
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_dict[axis]
    nrmse = 0.0
    batch_size = len(target)
    
    for b in range(batch_size):
        # Extract the axis data (assuming input is (batch, 303))
        pred_axis = pred[b].reshape(3, -1).T[:, axis_idx]
        targ_axis = target[b].reshape(3, -1).T[:, axis_idx]
        
        # Apply the same transformation as in evaluation script
        # This maintains consistency even though it's not standard inverse transform
        pred_axis = (pred_axis - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]
        targ_axis = (targ_axis - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]
        
        # Calculate RMSE in the double-normalized space
        rmse = torch.sqrt(torch.mean((pred_axis - targ_axis) ** 2))
        # Get range in the double-normalized space for this batch
        range_norm = (targ_axis.max() - targ_axis.min())
        if range_norm > 0:
            nrmse += 100 * rmse / range_norm
        else:
            nrmse += 0.0
    
    return nrmse / batch_size


def comprehensive_evaluation(model, loader, device, scaler, set_name="TEST", 
                           training_ranges=None, epoch=None, print_results=True):
    """
    Comprehensive evaluation matching your current metrics format.
    Uses the same double-normalization for consistency.
    """
    model.eval()
    
    # Use the known training ranges
    if training_ranges is None:
        training_ranges = {
            'X': 75.98,
            'Y': 27.80,
            'Z': 55.54
        }
    
    # Collect all predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)    # Shape: (n_trials, 303)
    all_targets = np.concatenate(all_targets, axis=0) # Shape: (n_trials, 303)
    
    # Reshape to separate axes: (n_trials, 101) per axis
    pred_X = all_preds[:, 0:101]
    pred_Y = all_preds[:, 101:202] 
    pred_Z = all_preds[:, 202:303]
    
    true_X = all_targets[:, 0:101]
    true_Y = all_targets[:, 101:202]
    true_Z = all_targets[:, 202:303]
    
    # Apply the same double-normalization as in evaluation script
    for idx, axis in enumerate(['X', 'Y', 'Z']):
        pred_data = locals()[f'pred_{axis}']
        true_data = locals()[f'true_{axis}']
        
        # Apply the transformation: (data - min_) / scale_
        pred_transformed = (pred_data - scaler.min_[idx]) / scaler.scale_[idx]
        true_transformed = (true_data - scaler.min_[idx]) / scaler.scale_[idx]
        
        # Update the variables
        locals()[f'pred_{axis}'] = pred_transformed
        locals()[f'true_{axis}'] = true_transformed
    
    axis_data = {
        'X': {'pred': pred_X, 'true': true_X, 'range': training_ranges['X']},
        'Y': {'pred': pred_Y, 'true': true_Y, 'range': training_ranges['Y']},
        'Z': {'pred': pred_Z, 'true': true_Z, 'range': training_ranges['Z']}
    }
    
    results = {}
    
    if print_results:
        if epoch is not None:
            print(f"\n{'='*80}")
            print(f"{set_name} SET EVALUATION - EPOCH {epoch}")
        else:
            print(f"\n{'='*80}")
            print(f"{set_name} SET EVALUATION (Final)")
        print(f"{'='*80}")
        
        # Print ranges header
        print("Training data ranges (normalization denominators):")
        print("-"*70)
        for axis in ['X', 'Y', 'Z']:
            range_val = axis_data[axis]['range']
            print(f"{axis}: range={range_val:.2f}°")
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
            # Note: This nRMSE uses training range in degrees, not the transformed range
            nrmse = 100 * rmse / train_range
            
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
        # This matches your evaluation: RMSE in transformed space / training range in degrees
        global_nrmse = 100 * global_rmse / train_range
        
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
            print(f"  Training range: {train_range:.2f}°")
            print(f"  Per-trial Corr: {per_trial_corrs.mean():.3f}±{per_trial_corrs.std():.3f}")
            print(f"  Global Corr: {global_corr:.3f}")
            print(f"  Global RMSE: {global_rmse:.3f} (transformed units)")
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
        print(f"{set_name} SUMMARY - " + (f"EPOCH {epoch}" if epoch is not None else "FINAL"))
        print(f"{'='*80}")
        print(f"Average global correlation: {avg_global_corr:.3f}")
        print(f"Average global RMSE: {avg_global_rmse:.3f} (transformed units)")
        print(f"Average global nRMSE: {avg_global_nrmse:.2f}%")
        print(f"Average SD ratio: {avg_sd_ratio:.3f}")
        
        if set_name == "TRAIN" or "TRAINING" in set_name:
            if avg_sd_ratio > 0.9:
                print("✅ SD ratio > 0.9: Model captures good variance")
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
    model.train()
    total_loss = 0
    x_err = y_err = z_err = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        x_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'x', scaler)
        y_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'y', scaler)
        z_err += nRMSE_Axis_TLPerbatch(out.detach(), y, 'z', scaler)
    n = len(loader.dataset)
    return (total_loss / n, x_err / n, y_err / n, z_err / n)


def evaluate(model, loader, criterion, device, scaler):
    model.eval()
    loss = 0
    x_err = y_err = z_err = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += criterion(out, y).item() * X.size(0)
            x_err += nRMSE_Axis_TLPerbatch(out, y, 'x', scaler)
            y_err += nRMSE_Axis_TLPerbatch(out, y, 'y', scaler)
            z_err += nRMSE_Axis_TLPerbatch(out, y, 'z', scaler)
    n = len(loader.dataset)
    return (loss / n, x_err / n, y_err / n, z_err / n)


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


def full_retrain():
    """Full retrain using optimal hyperparameters on complete train+val data."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    check_data_availability()
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 2: FULL RETRAIN WITH OPTIMAL HYPERPARAMETERS")
    print("="*80)
    print(f"Training for {FULL_RETRAIN_EPOCHS} epochs (no early stopping)")
    print(f"Using complete train+val data (no validation split)")
    print(f"Optimal hyperparameters: {BEST_PARAMS}")
    print("="*80)
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'step': 'full_retrain',
        'epochs': FULL_RETRAIN_EPOCHS,
        'best_params': BEST_PARAMS,
        'data_split': 'full_train_no_validation',
        'note': 'Step 2 of refinement ladder: full retrain with optimal params'
    }
    
    with open(join(RESULTS_DIR, f'full_retrain_config_{timestamp}.json'), 'w') as f:
        json.dump(config, f, indent=2)

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold} - FULL RETRAIN")
        print(f"{'='*60}")
        
        # Load scaler
        scaler_path = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
        scaler = load(open(scaler_path, 'rb'))
        
        # Load FULL training data (no train/val split)
        full_train_dataset = CNNDataset(DATASET_DIR, DATA_TYPE, 'train', fold)
        test_dataset = CNNDataset(DATASET_DIR, DATA_TYPE, 'test', fold)
        
        print(f"Full training set: {len(full_train_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
        
        # Create data loaders - use FULL training data
        # Fixed: Set drop_last=True for both to avoid BatchNorm issues
        train_loader = DataLoader(full_train_dataset, 
                                  batch_size=BEST_PARAMS['batch_size'], 
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=BEST_PARAMS['batch_size'], 
                                 shuffle=False, drop_last=True)

        # Initialize model with optimal parameters
        model = MultiScaleCNN(
            kernels=BEST_PARAMS['kernels'],
            channels=BEST_PARAMS['channels'],
            n_layers=BEST_PARAMS['n_layers'],
            pooling=BEST_PARAMS['pooling'],
            dropout_conv=BEST_PARAMS['dropout_conv'],
            dropout_fc=BEST_PARAMS['dropout_fc']
        ).to(device)
        
        print(f"Model parameters: {count_parameters(model):,}")
        
        # Training setup
        criterion = create_loss(BEST_PARAMS['loss'])
        optimizer = create_optimizer(
            model.parameters(), 
            BEST_PARAMS['optimizer'],
            BEST_PARAMS['lr'], 
            BEST_PARAMS['weight_decay']
        )
        
        # No scheduler for full retrain (was 'none' in optimal params anyway)
        scheduler = None

        # Setup logging
        log_dir = join(LOGS_DIR, f'fold_{fold}_full_retrain')
        os.makedirs(log_dir, exist_ok=True)
        writer_train = SummaryWriter(join(log_dir, 'train'))
        writer_test = SummaryWriter(join(log_dir, 'test'))

        # Training loop - FIXED 150 EPOCHS, NO EARLY STOPPING
        print(f"Training for {FULL_RETRAIN_EPOCHS} epochs...")
        
        for epoch in range(FULL_RETRAIN_EPOCHS):
            # Training step
            tr_loss, tr_x, tr_y, tr_z = train_epoch(
                model, train_loader, optimizer, criterion, 
                device, scaler, BEST_PARAMS['grad_clip']
            )
            
            # Log training metrics
            writer_train.add_scalar('loss', tr_loss, epoch)
            writer_train.add_scalar('nrmse_x', tr_x, epoch)
            writer_train.add_scalar('nrmse_y', tr_y, epoch)
            writer_train.add_scalar('nrmse_z', tr_z, epoch)
            writer_train.add_scalar('nrmse_mean', (tr_x + tr_y + tr_z) / 3, epoch)
            
            # Print progress and comprehensive evaluation every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"\nEpoch {epoch+1:3d}: Loss={tr_loss:.4f}, "
                      f"nRMSE: X={tr_x:.2f}%, Y={tr_y:.2f}%, Z={tr_z:.2f}%")
                
                # Comprehensive training evaluation
                train_results = comprehensive_evaluation(
                    model, train_loader, device, scaler, 
                    set_name="TRAINING", epoch=epoch+1, print_results=True
                )
                
                # Comprehensive test evaluation  
                test_results = comprehensive_evaluation(
                    model, test_loader, device, scaler,
                    set_name="TEST", epoch=epoch+1, print_results=True
                )
                
                # Log comprehensive metrics to tensorboard
                for axis in ['X', 'Y', 'Z']:
                    writer_train.add_scalar(f'comprehensive/corr_{axis.lower()}', 
                                          train_results[axis]['global_corr'], epoch)
                    writer_train.add_scalar(f'comprehensive/sd_ratio_{axis.lower()}', 
                                          train_results[axis]['sd_ratio'], epoch)
                    writer_test.add_scalar(f'comprehensive/corr_{axis.lower()}', 
                                         test_results[axis]['global_corr'], epoch)
                    writer_test.add_scalar(f'comprehensive/sd_ratio_{axis.lower()}', 
                                         test_results[axis]['sd_ratio'], epoch)
                
                # Log summary metrics
                writer_train.add_scalar('comprehensive/avg_sd_ratio', 
                                      train_results['summary']['avg_sd_ratio'], epoch)
                writer_test.add_scalar('comprehensive/avg_sd_ratio', 
                                     test_results['summary']['avg_sd_ratio'], epoch)

        # Final comprehensive evaluation
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION - FOLD {fold}")
        print(f"{'='*80}")
        
        final_train_results = comprehensive_evaluation(
            model, train_loader, device, scaler,
            set_name="FINAL TRAINING", print_results=True
        )
        
        final_test_results = comprehensive_evaluation(
            model, test_loader, device, scaler,
            set_name="FINAL TEST", print_results=True
        )
        
        # Save comprehensive results to JSON
        final_results = {
            'fold': fold,
            'epoch': FULL_RETRAIN_EPOCHS,
            'training': final_train_results,
            'test': final_test_results,
            'hyperparameters': BEST_PARAMS
        }
        
        results_file = join(RESULTS_DIR, f'fold_{fold}_comprehensive_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
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
            
            json.dump(convert_numpy(final_results), f, indent=2)
        
        # Close writers
        writer_train.close()
        writer_test.close()
        
        # Save model
        model_path = join(MODELS_DIR, f'fold_{fold}_full_retrain.pt')
        torch.save(model.state_dict(), model_path)
        
        full_model_path = join(MODELS_DIR, f'fold_{fold}_full_retrain_full.pt')
        torch.save(model, full_model_path)

    print("\n" + "="*80)
    print("FULL RETRAIN COMPLETED!")
    print("="*80)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")
    print("\nNext step: Run evaluation script to compare with original results,")
    print("then proceed to Step 3: Multi-seed ensemble if improvement is modest.")


if __name__ == '__main__':
    full_retrain()