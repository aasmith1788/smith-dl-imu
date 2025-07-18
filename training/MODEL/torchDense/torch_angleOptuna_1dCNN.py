# Multi-scale 1D CNN Optuna search script for IMU joint angle prediction

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
import optuna


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
# Configuration - UPDATED DATA PATHS
# ------------------------------
DATASET_NAME = 'IWALQQ_1st_correction'
DATA_TYPE = 'angle'

# FIXED: Updated to match the working script's data path structure
BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)

# Output directories
RESULTS_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\MultiScaleCNN_optuna'
MODELS_DIR = join(RESULTS_DIR, 'models')
LOGS_DIR = join(RESULTS_DIR, 'logs')

# Training parameters
N_TRIALS = 100
N_FOLDS = 5
MAX_EPOCHS = 50
MAX_PARAMETERS = 500_000

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Dataset directory: {DATASET_DIR}")


# ------------------------------
# Dataset utilities
# ------------------------------
class CNNDataset(Dataset):
    """Dataset that reshapes flattened IMU sequences to (42, 101).
    It also loads participant identifiers when available for
    participant-aware splitting."""

    def __init__(self, data_dir, data_type, sess, fold):
        data_file = join(data_dir, f"{fold}_fold_final_{sess}.npz")
        print(f"Loading data from: {data_file}")
        
        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data = np.load(data_file)
        X = np.squeeze(data[f"final_X_{sess}"]).astype(np.float32)
        Y = np.squeeze(data[f"final_Y_{data_type}_{sess}"]).astype(np.float32)
        X = X.reshape(-1, 42, 101)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        
        # Attempt to read participant IDs; default to index if missing
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


def get_train_val_indices(participants, val_ratio=0.2, random_state=42):
    """Split indices by participant so no subject leaks between sets."""
    df = pd.DataFrame({'pid': participants})
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio,
                            random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['pid']))
    return train_idx, val_idx


# ------------------------------
# Dataset caching
# ------------------------------
class DatasetCache:
    """Cache datasets to avoid reloading during hyperparameter search."""
    
    def __init__(self):
        self.cache = {}
        self.scalers = {}
    
    def get_datasets(self, dataset_dir, fold):
        """Get cached datasets or load them if not cached."""
        key = (dataset_dir, fold)
        if key not in self.cache:
            # FIXED: Load scaler with error checking
            scaler_path = join(dataset_dir, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl")
            print(f"Loading scaler from: {scaler_path}")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            self.scalers[fold] = load(open(scaler_path, 'rb'))
            
            # Load datasets
            full_dataset = CNNDataset(dataset_dir, DATA_TYPE, 'train', fold)
            test_dataset = CNNDataset(dataset_dir, DATA_TYPE, 'test', fold)
            
            # Get indices
            train_idx, val_idx = get_train_val_indices(full_dataset.participants)
            
            # Create subsets
            train_set = torch.utils.data.Subset(full_dataset, train_idx)
            val_set = torch.utils.data.Subset(full_dataset, val_idx)
            
            # Cache
            self.cache[key] = (train_set, val_set, test_dataset)
            print(f"Cached datasets for fold {fold}: {len(train_set)} train, {len(val_set)} val, {len(test_dataset)} test")
        
        return self.cache[key], self.scalers[fold]


# Global dataset cache
dataset_cache = DatasetCache()


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
            # Skip batch norm for single sample during training
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
                # Use SafeBatchNorm1d for conv layers
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
            # FIXED: Use regular BatchNorm1d for FC layers (2D input)
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
# Metrics
# ------------------------------

def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler):
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    axis = axis_dict[axis]
    nrmse = 0.0
    batch_size = len(target)
    for b in range(batch_size):
        pred_axis = pred[b].reshape(3, -1).T[:, axis]
        targ_axis = target[b].reshape(3, -1).T[:, axis]
        pred_axis = (pred_axis - scaler.min_[axis]) / scaler.scale_[axis]
        targ_axis = (targ_axis - scaler.min_[axis]) / scaler.scale_[axis]
        rmse = torch.sqrt(torch.mean((pred_axis - targ_axis) ** 2))
        nrmse += 100 * rmse / (targ_axis.max() - targ_axis.min())
    return nrmse


# ------------------------------
# Helper classes
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
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


# ------------------------------
# Optuna objective
# ------------------------------

def suggest_architecture(trial):
    kernels = trial.suggest_categorical('kernels',
                                        [[3, 5, 7], [3, 5], [5, 7, 9], [3, 7]])
    channels = trial.suggest_categorical('channels',
                                         [[8, 16, 32], [16, 32, 64],
                                          [32, 64, 128]])
    n_layers = trial.suggest_int('n_layers', 2, 4)
    pooling = trial.suggest_categorical('pooling',
                                       ['adaptive_avg', 'adaptive_max',
                                        'global_avg'])
    dropout_conv = trial.suggest_float('dropout_conv', 0.0, 0.5)
    dropout_fc = trial.suggest_float('dropout_fc', 0.0, 0.5)
    return kernels, channels, n_layers, pooling, dropout_conv, dropout_fc


def suggest_training(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer',
                                               ['adam', 'nadam', 'adamw', 'sgd'])
    loss_name = trial.suggest_categorical('loss', ['mse', 'mae', 'rmse', 'huber'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    scheduler = trial.suggest_categorical('scheduler', ['none', 'plateau'])
    patience = trial.suggest_int('patience', 10, 30)
    grad_clip = trial.suggest_float('grad_clip', 0.5, 5.0)
    return {
        'lr': lr,
        'optimizer': optimizer_name,
        'loss': loss_name,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'scheduler': scheduler,
        'patience': patience,
        'grad_clip': grad_clip,
    }


def check_data_availability():
    """Check if required data files exist before starting optimization."""
    print("Checking data availability...")
    print(f"Dataset directory: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    # Check for all required files
    missing_files = []
    for fold in range(N_FOLDS):
        # Check data files
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
    
    # List available files for verification
    print("Available files in dataset directory:")
    for file in sorted(os.listdir(DATASET_DIR)):
        if file.endswith(('.npz', '.pkl')):
            print(f"  - {file}")


def objective(trial):
    kernels, channels, n_layers, pooling, dconv, dfc = suggest_architecture(trial)
    train_params = suggest_training(trial)

    fold_losses = []

    for fold in range(N_FOLDS):
        # FIXED: Use the updated dataset directory
        # Get cached datasets
        (train_set, val_set, _), scaler = dataset_cache.get_datasets(DATASET_DIR, fold)
        
        # FIXED: Only drop_last for training
        train_loader = DataLoader(train_set, batch_size=train_params['batch_size'],
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=train_params['batch_size'],
                                shuffle=False, drop_last=False)

        model = MultiScaleCNN(kernels, channels, n_layers, pooling,
                              dconv, dfc).to(device)
        if count_parameters(model) > MAX_PARAMETERS:
            raise optuna.exceptions.TrialPruned()

        criterion = create_loss(train_params['loss'])
        optimizer = create_optimizer(model.parameters(), train_params['optimizer'],
                                     train_params['lr'], train_params['weight_decay'])
        scheduler = None
        if train_params['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7)
        early_stop = EarlyStopping(patience=train_params['patience'])

        for epoch in range(MAX_EPOCHS):
            train_metrics = train_epoch(model, train_loader, optimizer, criterion,
                                        device, scaler, train_params['grad_clip'])
            val_metrics = evaluate(model, val_loader, criterion, device, scaler)
            val_loss = val_metrics[0]
            if scheduler:
                scheduler.step(val_loss)
            if early_stop(val_loss):
                break
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_losses.append(val_loss)
        del model
        torch.cuda.empty_cache()

    return float(np.mean(fold_losses))


# ------------------------------
# Study execution and final training
# ------------------------------

def run_study():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check data availability first
    check_data_availability()
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Set seed for Optuna
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    print(f"Starting Optuna optimization with {N_TRIALS} trials...")
    study.optimize(objective, n_trials=N_TRIALS)

    with open(join(RESULTS_DIR, f'study_results_{timestamp}.json'), 'w') as f:
        json.dump({'best_params': study.best_params,
                   'best_value': study.best_value,
                   'n_trials': len(study.trials)}, f, indent=2)

    # Train final models using best hyperparameters
    print("Training final models with best hyperparameters...")
    best = study.best_params
    kernels, channels, n_layers, pooling, dconv, dfc = suggest_architecture(study.best_trial)
    train_p = suggest_training(study.best_trial)

    for fold in range(N_FOLDS):
        print(f"Training final model for fold {fold}...")
        
        # Get cached datasets
        (train_set, val_set, test_dataset), scaler = dataset_cache.get_datasets(DATASET_DIR, fold)
        
        # FIXED: Proper drop_last configuration
        train_loader = DataLoader(train_set, batch_size=train_p['batch_size'], 
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=train_p['batch_size'], 
                                shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=train_p['batch_size'], 
                                 shuffle=False, drop_last=False)

        model = MultiScaleCNN(kernels, channels, n_layers, pooling, dconv, dfc).to(device)
        criterion = create_loss(train_p['loss'])
        optimizer = create_optimizer(model.parameters(), train_p['optimizer'],
                                     train_p['lr'], train_p['weight_decay'])
        scheduler = None
        if train_p['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7)
        early_stop = EarlyStopping(patience=train_p['patience'])

        log_dir = join(LOGS_DIR, f'fold_{fold}')
        os.makedirs(log_dir, exist_ok=True)
        writer_train = SummaryWriter(join(log_dir, 'train'))
        writer_val = SummaryWriter(join(log_dir, 'val'))
        writer_test = SummaryWriter(join(log_dir, 'test'))
        best_state = None
        best_val = float('inf')

        for epoch in range(MAX_EPOCHS):
            tr_loss, tr_x, tr_y, tr_z = train_epoch(model, train_loader, optimizer,
                                                    criterion, device, scaler,
                                                    train_p['grad_clip'])
            val_loss, val_x, val_y, val_z = evaluate(model, val_loader, criterion,
                                                     device, scaler)
            if scheduler:
                scheduler.step(val_loss)
            
            # FIXED: Log all metrics, not just loss
            writer_train.add_scalar('loss', tr_loss, epoch)
            writer_train.add_scalar('nrmse_x', tr_x, epoch)
            writer_train.add_scalar('nrmse_y', tr_y, epoch)
            writer_train.add_scalar('nrmse_z', tr_z, epoch)
            writer_train.add_scalar('nrmse_mean', (tr_x + tr_y + tr_z) / 3, epoch)
            
            writer_val.add_scalar('loss', val_loss, epoch)
            writer_val.add_scalar('nrmse_x', val_x, epoch)
            writer_val.add_scalar('nrmse_y', val_y, epoch)
            writer_val.add_scalar('nrmse_z', val_z, epoch)
            writer_val.add_scalar('nrmse_mean', (val_x + val_y + val_z) / 3, epoch)
            
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
            if early_stop(val_loss):
                break

        if best_state:
            model.load_state_dict(best_state)
        test_loss, tx, ty, tz = evaluate(model, test_loader, criterion, device, scaler)
        
        # FIXED: Log all test metrics
        writer_test.add_scalar('loss', test_loss, 0)
        writer_test.add_scalar('nrmse_x', tx, 0)
        writer_test.add_scalar('nrmse_y', ty, 0)
        writer_test.add_scalar('nrmse_z', tz, 0)
        writer_test.add_scalar('nrmse_mean', (tx + ty + tz) / 3, 0)
        
        writer_train.close()
        writer_val.close()
        writer_test.close()
        
        # FIXED: Save model using torch.save instead of torch.jit.script
        # This avoids potential JIT scripting issues
        model_path = join(MODELS_DIR, f'fold_{fold}_best.pt')
        torch.save(model.state_dict(), model_path)
        
        # Also save the full model for easier loading
        full_model_path = join(MODELS_DIR, f'fold_{fold}_best_full.pt')
        torch.save(model, full_model_path)
        
        print(f"Saved model for fold {fold}: {model_path}")
        
        del model
        torch.cuda.empty_cache()

    print('Study finished. Best value:', study.best_value)
    print(f'Results saved to: {RESULTS_DIR}')
    print(f'Models saved to: {MODELS_DIR}')
    print(f'Logs saved to: {LOGS_DIR}')


if __name__ == '__main__':
    run_study()