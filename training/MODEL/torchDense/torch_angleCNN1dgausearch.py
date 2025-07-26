# Step 3.5: Gaussian noise level optimization with validation set
# Tests different noise levels to find optimal augmentation parameters

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
# Configuration - NOISE SEARCH PARAMETERS
# ------------------------------
# Different noise levels to test
NOISE_LEVELS_TO_TEST = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15]
VAL_RATIO = 0.2  # 20% for validation
N_SEEDS_NOISE_SEARCH = 2  # Reduced for faster search
NOISE_SEARCH_SEEDS = [42, 123]  # Use fewer seeds for noise search
NOISE_SEARCH_EPOCHS = 15 # Shorter training for search

DATASET_NAME = 'IWALQQ_1st_correction'
DATA_TYPE = 'angle'

BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)

# Output directories for noise search
RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\MultiScaleCNN_noise_search'
MODELS_DIR = join(RESULTS_DIR, 'models')
LOGS_DIR = join(RESULTS_DIR, 'logs')
NOISE_RESULTS_DIR = join(RESULTS_DIR, 'noise_search_results')

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
N_FOLDS = 5
MAX_PARAMETERS = 500_000

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
# Helper function to get underlying dataset
# ------------------------------
def get_base_dataset(loader_dataset):
    """Get the underlying CNNDataset from a potentially wrapped dataset."""
    if isinstance(loader_dataset, torch.utils.data.Subset):
        return loader_dataset.dataset
    return loader_dataset


# ------------------------------
# Dataset utilities with configurable Gaussian noise
# ------------------------------
class CNNDataset(Dataset):
    """Dataset that reshapes flattened IMU sequences to (42, 101) with configurable Gaussian noise."""

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
        self.training = False  # Default to eval mode
        
        # Initialize channel noise stds (will be calculated when needed)
        self.channel_noise_stds = torch.zeros(42)
        
        # Calculate initial noise levels if noise is enabled
        if self.apply_noise and self.noise_std_fraction > 0:
            self._calculate_channel_noise_stds()
        
        # Load participant IDs if available
        pid_key = f"participant_{sess}"
        if pid_key in data:
            self.participants = data[pid_key]
        elif f"PID_{sess}" in data:
            self.participants = data[f"PID_{sess}"]
        else:
            self.participants = np.arange(len(X))
        
        print(f"Loaded {len(self.X)} samples for {sess} set, fold {fold}")

    def _calculate_channel_noise_stds(self):
        """Calculate channel-wise noise standard deviations."""
        if self.noise_std_fraction > 0:
            for ch in range(42):
                channel_data = self.X[:, ch, :].flatten()
                channel_std = channel_data.std().item()
                self.channel_noise_stds[ch] = self.noise_std_fraction * channel_std
            
            print(f"  Noise augmentation: std fraction={self.noise_std_fraction:.3f}")
            print(f"  Channel noise STDs range: [{self.channel_noise_stds.min():.4f}, {self.channel_noise_stds.max():.4f}]")
        else:
            self.channel_noise_stds = torch.zeros(42)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # Shape: (42, 101)
        y = self.Y[idx]
        
        # Apply Gaussian noise during training only
        if self.apply_noise and self.training and self.noise_std_fraction > 0:
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

    def update_noise_level(self, new_noise_fraction):
        """Update noise level during search."""
        self.noise_std_fraction = new_noise_fraction
        # Recalculate channel noise standard deviations
        self._calculate_channel_noise_stds()


def get_train_val_indices(participants, val_ratio=0.2, random_state=42):
    """Split indices by participant so no subject leaks between sets."""
    df = pd.DataFrame({'pid': participants})
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['pid']))
    return train_idx, val_idx


# ------------------------------
# Model definition (same as before)
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


class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))


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


def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler):
    """Calculate nRMSE for a specific axis."""
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_dict[axis]
    nrmse = 0.0
    batch_size = len(target)
    
    for b in range(batch_size):
        pred_axis = pred[b].reshape(3, -1).T[:, axis_idx]
        targ_axis = target[b].reshape(3, -1).T[:, axis_idx]
        
        pred_axis = (pred_axis - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]
        targ_axis = (targ_axis - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]
        
        rmse = torch.sqrt(torch.mean((pred_axis - targ_axis) ** 2))
        range_norm = (targ_axis.max() - targ_axis.min())
        if range_norm > 0:
            nrmse += 100 * rmse / range_norm
        else:
            nrmse += 0.0
    
    return nrmse / batch_size


def train_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip):
    """Train one epoch with Gaussian noise applied to inputs."""
    model.train()
    
    # Enable training mode in dataset to apply noise
    base_dataset = get_base_dataset(loader.dataset)
    if hasattr(base_dataset, 'set_training'):
        base_dataset.set_training(True)
    
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
    
    # Disable training mode after epoch
    if hasattr(base_dataset, 'set_training'):
        base_dataset.set_training(False)
    
    n = len(loader.dataset)
    return (total_loss / n, x_err / n, y_err / n, z_err / n)


def evaluate(model, loader, criterion, device, scaler):
    """Evaluate model without noise."""
    model.eval()
    
    # Ensure no noise during evaluation
    base_dataset = get_base_dataset(loader.dataset)
    if hasattr(base_dataset, 'set_training'):
        base_dataset.set_training(False)
    
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


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_single_model_with_noise(fold, noise_level, seed, train_loader, val_loader, scaler):
    """Train a single model with a specific noise level."""
    
    # Set seed for this model
    set_seed(seed)
    
    # Update noise level in dataset - USE HELPER TO GET BASE DATASET
    base_train_dataset = get_base_dataset(train_loader.dataset)
    if hasattr(base_train_dataset, 'update_noise_level'):
        base_train_dataset.update_noise_level(noise_level)
        print(f"        Updated noise level to {noise_level:.3f}")
    
    # Initialize model
    model = MultiScaleCNN(
        kernels=BEST_PARAMS['kernels'],
        channels=BEST_PARAMS['channels'],
        n_layers=BEST_PARAMS['n_layers'],
        pooling=BEST_PARAMS['pooling'],
        dropout_conv=BEST_PARAMS['dropout_conv'],
        dropout_fc=BEST_PARAMS['dropout_fc']
    ).to(device)
    
    # Training setup
    criterion = create_loss(BEST_PARAMS['loss'])
    optimizer = create_optimizer(
        model.parameters(), 
        BEST_PARAMS['optimizer'],
        BEST_PARAMS['lr'], 
        BEST_PARAMS['weight_decay']
    )
    
    early_stop = EarlyStopping(patience=5, min_delta=0.001)  # Shorter patience for search
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_gap = float('inf')
    final_train_loss = float('inf')
    final_val_loss = float('inf')

    # Training loop
    for epoch in range(NOISE_SEARCH_EPOCHS):
        # Training step (with noise)
        tr_loss, tr_x, tr_y, tr_z = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, scaler, BEST_PARAMS['grad_clip']
        )
        
        # Validation step (without noise)
        val_loss, val_x, val_y, val_z = evaluate(
            model, val_loader, criterion, device, scaler
        )
        
        # Calculate train-val gap
        gap = val_loss - tr_loss
        
        # Track best validation performance AND final performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = tr_loss
            best_gap = gap
        
        # Always update final performance (last epoch)
        final_train_loss = tr_loss
        final_val_loss = val_loss
        
        # Early stopping based on validation loss
        if early_stop(val_loss):
            break
    
    # Return both best performance and final performance
    return {
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'best_gap': best_gap,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_gap': final_val_loss - final_train_loss
    }


def search_optimal_noise_level():
    """Search for optimal Gaussian noise level using validation set."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(NOISE_RESULTS_DIR, exist_ok=True)
    
    print("="*80)
    print("GAUSSIAN NOISE LEVEL OPTIMIZATION WITH VALIDATION SET")
    print("="*80)
    print(f"Testing noise levels: {NOISE_LEVELS_TO_TEST}")
    print(f"Validation ratio: {VAL_RATIO}")
    print(f"Seeds per noise level: {NOISE_SEARCH_SEEDS}")
    print(f"Training epochs per model: {NOISE_SEARCH_EPOCHS}")
    print("="*80)
    
    # Store results for all folds and noise levels
    all_results = {}
    
    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold} - NOISE LEVEL SEARCH")
        print(f"{'='*60}")
        
        # Load scaler
        scaler_path = join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
        scaler = load(open(scaler_path, 'rb'))
        
        # Load full training data
        full_train_dataset = CNNDataset(
            DATASET_DIR, DATA_TYPE, 'train', fold,
            apply_noise=True,  # Will be controlled by noise level
            noise_std_fraction=0.0,  # Will be updated for each test
            scaler=scaler
        )
        
        # Split into train/val by participant
        train_idx, val_idx = get_train_val_indices(
            full_train_dataset.participants, 
            val_ratio=VAL_RATIO, 
            random_state=42
        )
        
        # Create subsets
        train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_train_dataset, val_idx)
        
        print(f"Train samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        # Test each noise level
        fold_results = {}
        
        for noise_level in NOISE_LEVELS_TO_TEST:
            print(f"\n  Testing noise level: {noise_level:.3f}")
            
            # Create data loaders
            train_loader = DataLoader(train_subset, 
                                      batch_size=BEST_PARAMS['batch_size'], 
                                      shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, 
                                    batch_size=BEST_PARAMS['batch_size'], 
                                    shuffle=False, drop_last=False)
            
            # Train multiple models with different seeds for this noise level
            seed_results = []
            
            for seed_idx, seed in enumerate(NOISE_SEARCH_SEEDS):
                print(f"    Seed {seed} ({seed_idx+1}/{len(NOISE_SEARCH_SEEDS)})", end=" ")
                
                result = train_single_model_with_noise(
                    fold, noise_level, seed, train_loader, val_loader, scaler
                )
                
                seed_results.append({
                    'seed': seed,
                    'best_val_loss': result['best_val_loss'],
                    'best_train_loss': result['best_train_loss'],
                    'best_gap': result['best_gap'],
                    'final_val_loss': result['final_val_loss'],
                    'final_train_loss': result['final_train_loss'],
                    'final_gap': result['final_gap']
                })
                
                print(f"- Val Loss: {result['best_val_loss']:.4f}, Gap: {result['best_gap']:.4f}")
            
            # Calculate averages across seeds for this noise level
            avg_val_loss = np.mean([r['best_val_loss'] for r in seed_results])
            avg_train_loss = np.mean([r['best_train_loss'] for r in seed_results])
            avg_gap = np.mean([r['best_gap'] for r in seed_results])
            avg_final_gap = np.mean([r['final_gap'] for r in seed_results])
            
            std_val_loss = np.std([r['best_val_loss'] for r in seed_results])
            std_train_loss = np.std([r['best_train_loss'] for r in seed_results])
            std_gap = np.std([r['best_gap'] for r in seed_results])
            std_final_gap = np.std([r['final_gap'] for r in seed_results])
            
            fold_results[f'noise_{noise_level:.3f}'] = {
                'noise_level': noise_level,
                'avg_val_loss': avg_val_loss,
                'avg_train_loss': avg_train_loss,
                'avg_gap': avg_gap,
                'avg_final_gap': avg_final_gap,
                'std_val_loss': std_val_loss,
                'std_train_loss': std_train_loss,
                'std_gap': std_gap,
                'std_final_gap': std_final_gap,
                'seed_results': seed_results
            }
            
            print(f"    Average: Val Loss={avg_val_loss:.4f}±{std_val_loss:.4f}, "
                  f"Train-Val Gap={avg_gap:.4f}±{std_gap:.4f}")
        
        # Find best noise level for this fold based on SMALLEST GAP
        best_gap_key = min(fold_results.keys(), 
                          key=lambda k: fold_results[k]['avg_gap'])
        best_gap_noise = fold_results[best_gap_key]['noise_level']
        best_gap_performance = fold_results[best_gap_key]
        
        # Also find best validation loss
        best_val_key = min(fold_results.keys(), 
                          key=lambda k: fold_results[k]['avg_val_loss'])
        best_val_noise = fold_results[best_val_key]['noise_level']
        best_val_performance = fold_results[best_val_key]
        
        print(f"\n  FOLD {fold} RESULTS:")
        print(f"    Best Gap (Generalization): Noise={best_gap_noise:.3f}, Gap={best_gap_performance['avg_gap']:.4f}±{best_gap_performance['std_gap']:.4f}")
        print(f"    Best Val Loss: Noise={best_val_noise:.3f}, Loss={best_val_performance['avg_val_loss']:.4f}±{best_val_performance['std_val_loss']:.4f}")
        
        all_results[f'fold_{fold}'] = {
            'fold': fold,
            'best_gap_noise_level': best_gap_noise,
            'best_gap_performance': best_gap_performance,
            'best_val_noise_level': best_val_noise,
            'best_val_performance': best_val_performance,
            'all_noise_results': fold_results
        }
        
        # Save fold results
        fold_file = join(NOISE_RESULTS_DIR, f'fold_{fold}_noise_search.json')
        with open(fold_file, 'w') as f:
            json.dump(convert_numpy(all_results[f'fold_{fold}']), f, indent=2)
    
    # Calculate overall best noise level across all folds
    print(f"\n{'='*80}")
    print("NOISE LEVEL SEARCH SUMMARY")
    print(f"{'='*80}")
    
    # Find best noise levels across all folds for both criteria
    all_best_gap_noise = [all_results[f'fold_{i}']['best_gap_noise_level'] for i in range(N_FOLDS)]
    all_best_val_noise = [all_results[f'fold_{i}']['best_val_noise_level'] for i in range(N_FOLDS)]
    
    avg_best_gap_noise = np.mean(all_best_gap_noise)
    avg_best_val_noise = np.mean(all_best_val_noise)
    
    print(f"Best gap noise levels per fold: {all_best_gap_noise}")
    print(f"Best val loss noise levels per fold: {all_best_val_noise}")
    print(f"Average best gap noise: {avg_best_gap_noise:.3f}")
    print(f"Average best val noise: {avg_best_val_noise:.3f}")
    
    # Calculate average performance for each noise level across folds
    noise_level_summary = {}
    for noise_level in NOISE_LEVELS_TO_TEST:
        noise_key = f'noise_{noise_level:.3f}'
        val_losses = []
        train_losses = []
        gaps = []
        
        for fold in range(N_FOLDS):
            if noise_key in all_results[f'fold_{fold}']['all_noise_results']:
                result = all_results[f'fold_{fold}']['all_noise_results'][noise_key]
                val_losses.append(result['avg_val_loss'])
                train_losses.append(result['avg_train_loss'])
                gaps.append(result['avg_gap'])
        
        if val_losses:
            noise_level_summary[noise_level] = {
                'avg_val_loss_across_folds': np.mean(val_losses),
                'std_val_loss_across_folds': np.std(val_losses),
                'avg_train_loss_across_folds': np.mean(train_losses),
                'std_train_loss_across_folds': np.std(train_losses),
                'avg_gap_across_folds': np.mean(gaps),
                'std_gap_across_folds': np.std(gaps),
                'val_losses': val_losses,
                'train_losses': train_losses,
                'gaps': gaps
            }
    
    # Find globally optimal noise levels for both criteria
    best_global_gap_noise = min(noise_level_summary.keys(), 
                               key=lambda k: noise_level_summary[k]['avg_gap_across_folds'])
    best_global_val_noise = min(noise_level_summary.keys(), 
                               key=lambda k: noise_level_summary[k]['avg_val_loss_across_folds'])
    
    print(f"\nNoise level performance summary (averaged across {N_FOLDS} folds):")
    print("-" * 90)
    print(f"{'Noise':<8} {'Val Loss':<15} {'Train Loss':<15} {'Gap':<15} {'Best For'}")
    print("-" * 90)
    
    for noise_level in sorted(noise_level_summary.keys()):
        summary = noise_level_summary[noise_level]
        val_loss = summary['avg_val_loss_across_folds']
        val_std = summary['std_val_loss_across_folds']
        train_loss = summary['avg_train_loss_across_folds']
        train_std = summary['std_train_loss_across_folds']
        gap = summary['avg_gap_across_folds']
        gap_std = summary['std_gap_across_folds']
        
        markers = []
        if noise_level == best_global_gap_noise:
            markers.append("Gap")
        if noise_level == best_global_val_noise:
            markers.append("Val Loss")
        marker_str = ", ".join(markers) if markers else ""
        if markers:
            marker_str = f"✅ {marker_str}"
        
        print(f"{noise_level:<8.3f} {val_loss:.4f}±{val_std:.4f}  {train_loss:.4f}±{train_std:.4f}  {gap:.4f}±{gap_std:.4f}  {marker_str}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   Best Generalization (smallest gap): {best_global_gap_noise:.3f}")
    gap_perf = noise_level_summary[best_global_gap_noise]
    print(f"      Average gap: {gap_perf['avg_gap_across_folds']:.4f}±{gap_perf['std_gap_across_folds']:.4f}")
    print(f"      Validation loss: {gap_perf['avg_val_loss_across_folds']:.4f}±{gap_perf['std_val_loss_across_folds']:.4f}")
    
    print(f"   Best Validation Loss: {best_global_val_noise:.3f}")
    val_perf = noise_level_summary[best_global_val_noise]
    print(f"      Validation loss: {val_perf['avg_val_loss_across_folds']:.4f}±{val_perf['std_val_loss_across_folds']:.4f}")
    print(f"      Gap: {val_perf['avg_gap_across_folds']:.4f}±{val_perf['std_gap_across_folds']:.4f}")
    
    # Decide which to recommend
    if best_global_gap_noise == best_global_val_noise:
        recommended_noise = best_global_gap_noise
        print(f"\n🏆 FINAL RECOMMENDATION: {recommended_noise:.3f}")
        print("   This noise level optimizes both generalization and validation performance!")
    else:
        # Check if gap difference is significant
        gap_diff = gap_perf['avg_gap_across_folds'] - val_perf['avg_gap_across_folds']
        val_diff = val_perf['avg_val_loss_across_folds'] - gap_perf['avg_val_loss_across_folds']
        
        if abs(gap_diff) < 0.001:  # Very small gap difference
            recommended_noise = best_global_val_noise
            print(f"\n🏆 FINAL RECOMMENDATION: {recommended_noise:.3f}")
            print("   Gap difference is minimal, choosing best validation loss.")
        elif abs(val_diff) < 0.001:  # Very small validation difference  
            recommended_noise = best_global_gap_noise
            print(f"\n🏆 FINAL RECOMMENDATION: {recommended_noise:.3f}")
            print("   Validation loss difference is minimal, choosing best generalization.")
        else:
            recommended_noise = best_global_gap_noise  # Prefer generalization
            print(f"\n🏆 FINAL RECOMMENDATION: {recommended_noise:.3f}")
            print("   Prioritizing generalization (smallest train-val gap) over absolute performance.")
    
    # Save comprehensive summary
    summary = {
        'timestamp': timestamp,
        'search_parameters': {
            'noise_levels_tested': NOISE_LEVELS_TO_TEST,
            'validation_ratio': VAL_RATIO,
            'seeds_per_level': NOISE_SEARCH_SEEDS,
            'epochs_per_model': NOISE_SEARCH_EPOCHS,
            'n_folds': N_FOLDS
        },
        'results': {
            'recommended_noise_level': recommended_noise,
            'best_gap_noise_level': best_global_gap_noise,
            'best_val_noise_level': best_global_val_noise,
            'noise_level_summary': noise_level_summary,
            'fold_results': all_results
        },
        'hyperparameters_used': BEST_PARAMS
    }
    
    summary_file = join(RESULTS_DIR, f'noise_search_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2)
    
    print(f"\n{'='*80}")
    print("NOISE LEVEL SEARCH COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Detailed results: {NOISE_RESULTS_DIR}")
    print(f"Summary file: {summary_file}")
    print(f"\nNext steps:")
    print(f"1. Use noise level {recommended_noise:.3f} for final ensemble training")
    print(f"2. Monitor both validation loss AND train-val gap in final training")
    print(f"3. If overfitting persists, consider the gap-optimized level: {best_global_gap_noise:.3f}")
    
    return recommended_noise, summary


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
        print("Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        raise FileNotFoundError("Some required data files are missing")
    
    print("All required data files found!")


# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    print("Starting Gaussian Noise Level Optimization...")
    
    # Check data availability first
    check_data_availability()
    
    # Run the noise level search
    try:
        optimal_noise, search_results = search_optimal_noise_level()
        print(f"\n🎉 SUCCESS! Optimal noise level found: {optimal_noise:.3f}")
        
    except Exception as e:
        print(f"\n❌ ERROR during noise search: {str(e)}")
        import traceback
        traceback.print_exc()
        raise