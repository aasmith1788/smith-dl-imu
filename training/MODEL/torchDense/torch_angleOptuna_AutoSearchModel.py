#!/usr/bin/env python3
"""
Automated Neural Architecture Search with Optuna for IMU Angle Prediction

This script uses Optuna to automatically search for optimal neural network architectures
and hyperparameters. It includes comprehensive TensorBoard logging to detect overfitting
and monitor training dynamics.

Features:
- Automated architecture search (layer sizes, activation functions, etc.)
- Hyperparameter optimization (learning rate, batch size, optimizer, etc.)
- Comprehensive TensorBoard logging for overfitting detection
- Cross-validation with early stopping and pruning
- Model saving and evaluation
"""

import subprocess
import sys
import os
import datetime
import json
from os.path import join
from pickle import load
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna

# Install required packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "torch", "torchvision", "torchaudio", "numpy", "tqdm", 
    "tensorboard", "optuna", "optuna-dashboard"
]

print("Installing required packages...")
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

print("All packages ready!\n")

# Configuration
DATASET_NAME = 'IWALQQ_1st_correction'
DATA_TYPE = 'angle'
N_TRIALS = 100
N_FOLDS = 5
MAX_EPOCHS_TRIAL = 100  # Epochs per trial (shorter for faster search)
MAX_EPOCHS_FINAL = 500  # Epochs for final model training
PRUNING_INTERVAL = 10

# Early stopping patience settings
TRIAL_EARLY_STOPPING_PATIENCE = 30   # Shorter for trials (faster optimization)
FINAL_EARLY_STOPPING_PATIENCE = 110  # Original value for final training (better performance)

# Paths
BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)
OUTPUT_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\Optuna_AutoSearch'
MODELS_DIR = join(OUTPUT_DIR, 'models')
LOGS_DIR = join(OUTPUT_DIR, 'logs')
OPTUNA_DIR = join(OUTPUT_DIR, 'optuna_studies')

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

# Create output directories
ensure_dir(OUTPUT_DIR)
ensure_dir(MODELS_DIR)
ensure_dir(LOGS_DIR)
ensure_dir(OPTUNA_DIR)

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

def init_weights(m):
    """Initialize weights using He initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class ConfigurableModel(nn.Module):
    """Configurable neural network architecture"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input/output dimensions
        input_dim = 4242
        output_dim = 303
        
        # Build layers
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Hidden layers
        prev_dim = input_dim
        for i in range(config['n_layers']):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, config['hidden_dims'][i]))
            
            # Batch normalization
            if config['use_batch_norm']:
                self.batch_norms.append(nn.BatchNorm1d(config['hidden_dims'][i]))
            else:
                self.batch_norms.append(None)
            
            # Dropout
            self.dropouts.append(nn.Dropout(config['dropout_rates'][i]))
            
            prev_dim = config['hidden_dims'][i]
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.flatten(x)
        
        for i in range(self.config['n_layers']):
            x = self.layers[i](x)
            
            if self.config['use_batch_norm'] and self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            
            # Activation function
            if self.config['activation'] == 'relu':
                x = F.relu(x)
            elif self.config['activation'] == 'gelu':
                x = F.gelu(x)
            elif self.config['activation'] == 'swish':
                x = F.silu(x)
            elif self.config['activation'] == 'tanh':
                x = F.tanh(x)
            elif self.config['activation'] == 'leaky_relu':
                x = F.leaky_relu(x, 0.1)
            
            x = self.dropouts[i](x)
        
        return self.output_layer(x)

class RMSELoss(nn.Module):
    """Root Mean Square Error loss"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)

class Dataset(torch.utils.data.Dataset):
    """Custom dataset for IMU data"""
    def __init__(self, dataset_dir, data_type, session, fold):
        self.data_type = data_type
        self.session = session
        
        # Load data
        data_file = join(dataset_dir, f"{fold}_fold_final_{session}.npz")
        data = np.load(data_file)
        
        self.X = torch.from_numpy(data[f'final_X_{session}']).squeeze().float()
        self.Y = torch.from_numpy(data[f'final_Y_{data_type}_{session}']).squeeze().float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def create_loss_function(loss_type):
    """Create loss function based on type"""
    if loss_type == 'rmse':
        return RMSELoss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def create_optimizer(model, optimizer_type, lr, weight_decay):
    """Create optimizer based on type"""
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'nadam':
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def nRMSE_axis_batch(pred, target, axis, scaler):
    """Calculate normalized RMSE for specific axis"""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis]
    
    batch_nrmse = 0
    batch_size = len(target)
    
    for i in range(batch_size):
        # Reshape to [timesteps, 3] and extract axis
        pred_axis = pred[i].view(3, -1).t()[:, axis_idx]
        target_axis = target[i].view(3, -1).t()[:, axis_idx]
        
        # Denormalize
        pred_axis = (pred_axis - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]
        target_axis = (target_axis - scaler.min_[axis_idx]) / scaler.scale_[axis_idx]
        
        # Calculate nRMSE
        rmse = torch.sqrt(torch.mean((pred_axis - target_axis) ** 2))
        range_val = torch.max(target_axis) - torch.min(target_axis)
        nrmse = 100 * rmse / range_val
        
        batch_nrmse += nrmse.item()
    
    return batch_nrmse

def evaluate(model, data_loader, criterion, scaler):
    """Evaluate model and return loss and per-axis nRMSE"""
    model.eval()
    total_loss = 0
    samples = 0
    x_nrmse = 0
    y_nrmse = 0
    z_nrmse = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            samples += data.size(0)
            x_nrmse += nRMSE_axis_batch(output, target, 'x', scaler)
            y_nrmse += nRMSE_axis_batch(output, target, 'y', scaler)
            z_nrmse += nRMSE_axis_batch(output, target, 'z', scaler)

    total_loss /= samples
    x_nrmse /= samples
    y_nrmse /= samples
    z_nrmse /= samples

    return total_loss, x_nrmse, y_nrmse, z_nrmse

def suggest_architecture(trial):
    """Suggest architecture hyperparameters"""
    # Number of layers
    n_layers = trial.suggest_int('n_layers', 2, 6)
    
    # Architecture pattern
    pattern = trial.suggest_categorical('arch_pattern', ['decreasing', 'increasing', 'pyramid', 'uniform'])
    
    # Layer size range
    min_size = trial.suggest_int('min_layer_size', 512, 2048)
    max_size = trial.suggest_int('max_layer_size', 2048, 8192)
    
    if min_size > max_size:
        min_size, max_size = max_size, min_size
    
    # Generate layer sizes based on pattern
    if pattern == 'decreasing':
        sizes = np.linspace(max_size, min_size, n_layers, dtype=int)
    elif pattern == 'increasing':
        sizes = np.linspace(min_size, max_size, n_layers, dtype=int)
    elif pattern == 'pyramid':
        mid = n_layers // 2
        up = np.linspace(min_size, max_size, mid + 1, dtype=int)
        down = np.linspace(max_size, min_size, n_layers - mid, dtype=int)[1:]
        sizes = np.concatenate([up, down])
    else:  # uniform
        size = trial.suggest_int('uniform_size', min_size, max_size)
        sizes = np.full(n_layers, size)
    
    # Dropout rates
    dropout_rates = []
    for i in range(n_layers):
        dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.1, 0.7))
    
    return {
        'n_layers': n_layers,
        'hidden_dims': sizes.tolist(),
        'dropout_rates': dropout_rates,
        'use_batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish', 'tanh', 'leaky_relu'])
    }

def suggest_training_params(trial):
    """Suggest training hyperparameters"""
    return {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'nadam', 'adamw', 'sgd']),
        'loss_function': trial.suggest_categorical('loss_function', ['rmse', 'mse', 'mae', 'huber']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'use_scheduler': trial.suggest_categorical('use_scheduler', [True, False]),
        'scheduler_patience': trial.suggest_int('scheduler_patience', 5, 25),
        'scheduler_factor': trial.suggest_float('scheduler_factor', 0.1, 0.8),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 20, 40),  # Range around trial patience
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 5.0)
    }

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                      early_stopping, scaler, max_epochs, trial=None, fold=None,
                      log_dir=None, is_final=False):
    """Train and evaluate model with comprehensive logging"""
    
    # Setup TensorBoard logging
    if log_dir:
        train_writer = SummaryWriter(join(log_dir, 'train'))
        val_writer = SummaryWriter(join(log_dir, 'val'))
        
        # Add model graph
        dummy_input = torch.randn(1, 4242).to(device)
        train_writer.add_graph(model, dummy_input)
    else:
        train_writer = val_writer = None
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        train_x_nrmse = 0
        train_y_nrmse = 0
        train_z_nrmse = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', leave=False)
        
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if hasattr(early_stopping, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), early_stopping.max_grad_norm)
            
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            train_samples += data.size(0)
            
            # Calculate nRMSE
            train_x_nrmse += nRMSE_axis_batch(output, target, 'x', scaler)
            train_y_nrmse += nRMSE_axis_batch(output, target, 'y', scaler)
            train_z_nrmse += nRMSE_axis_batch(output, target, 'z', scaler)
            
            train_bar.set_postfix({'loss': loss.item()})
        
        # Average training metrics
        train_loss /= train_samples
        train_x_nrmse /= train_samples
        train_y_nrmse /= train_samples
        train_z_nrmse /= train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        val_x_nrmse = 0
        val_y_nrmse = 0
        val_z_nrmse = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                val_samples += data.size(0)
                
                val_x_nrmse += nRMSE_axis_batch(output, target, 'x', scaler)
                val_y_nrmse += nRMSE_axis_batch(output, target, 'y', scaler)
                val_z_nrmse += nRMSE_axis_batch(output, target, 'z', scaler)
        
        # Average validation metrics
        val_loss /= val_samples
        val_x_nrmse /= val_samples
        val_y_nrmse /= val_samples
        val_z_nrmse /= val_samples
        
        # TensorBoard logging
        if train_writer and val_writer:
            # Training metrics
            train_writer.add_scalar('Loss', train_loss, epoch)
            train_writer.add_scalar('X_nRMSE', train_x_nrmse, epoch)
            train_writer.add_scalar('Y_nRMSE', train_y_nrmse, epoch)
            train_writer.add_scalar('Z_nRMSE', train_z_nrmse, epoch)
            train_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Validation metrics
            val_writer.add_scalar('Loss', val_loss, epoch)
            val_writer.add_scalar('X_nRMSE', val_x_nrmse, epoch)
            val_writer.add_scalar('Y_nRMSE', val_y_nrmse, epoch)
            val_writer.add_scalar('Z_nRMSE', val_z_nrmse, epoch)
            
            # Overfitting detection
            overfitting_gap = train_loss - val_loss
            train_writer.add_scalar('Overfitting_Gap', overfitting_gap, epoch)
            val_writer.add_scalar('Overfitting_Gap', overfitting_gap, epoch)
            
            # Gradient norm
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            train_writer.add_scalar('Gradient_Norm', total_grad_norm, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Pruning for Optuna trials
        if trial and epoch % PRUNING_INTERVAL == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                if train_writer:
                    train_writer.close()
                if val_writer:
                    val_writer.close()
                raise optuna.exceptions.TrialPruned()
        
        # Progress logging
        if is_final and epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'  Train nRMSE - X: {train_x_nrmse:.4f}, Y: {train_y_nrmse:.4f}, Z: {train_z_nrmse:.4f}')
            print(f'  Val nRMSE - X: {val_x_nrmse:.4f}, Y: {val_y_nrmse:.4f}, Z: {val_z_nrmse:.4f}')
            
            overfitting_gap = train_loss - val_loss
            print(f'  Overfitting Gap: {overfitting_gap:.4f}')
            if overfitting_gap > 0.02:
                print(f'  ⚠️  WARNING: Potential overfitting detected!')
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Close writers
    if train_writer:
        train_writer.close()
    if val_writer:
        val_writer.close()

    return best_val_loss, best_model_state

def objective(trial):
    """Optuna objective function"""
    
    # Get hyperparameters
    arch_config = suggest_architecture(trial)
    train_config = suggest_training_params(trial)
    
    print(f"\nTrial {trial.number}:")
    print(f"  Architecture: {arch_config['n_layers']} layers, {arch_config['hidden_dims']}")
    print(f"  Training: lr={train_config['learning_rate']:.2e}, batch={train_config['batch_size']}")
    
    fold_scores = []
    
    # Cross-validation
    for fold in range(N_FOLDS):
        print(f'  Fold {fold+1}/{N_FOLDS}')
        
        # Create model
        model = ConfigurableModel(arch_config).to(device)
        
        # Load full training dataset and split into train/validation
        full_dataset = Dataset(DATASET_DIR, DATA_TYPE, 'train', fold)
        val_ratio = 0.2
        train_size = int(len(full_dataset) * (1 - val_ratio))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
        
        # Create optimizer and loss
        criterion = create_loss_function(train_config['loss_function'])
        optimizer = create_optimizer(model, train_config['optimizer'], 
                                   train_config['learning_rate'], train_config['weight_decay'])
        
        # Create scheduler
        scheduler = None
        if train_config['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=train_config['scheduler_factor'],
                patience=train_config['scheduler_patience'], min_lr=1e-7
            )
        
        # Early stopping (use trial patience for optimization)
        early_stopping = EarlyStopping(
            patience=TRIAL_EARLY_STOPPING_PATIENCE,  # Use shorter patience for trials
            min_delta=0.001
        )
        early_stopping.max_grad_norm = train_config['max_grad_norm']
        
        # Load scaler
        scaler = load(open(join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl"), 'rb'))
        
        # Setup logging
        log_dir = join(LOGS_DIR, 'trials', f'trial_{trial.number}', f'fold_{fold}')
        ensure_dir(log_dir)
        
        # Train and evaluate on training/validation splits
        best_val_loss, _ = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            early_stopping, scaler, MAX_EPOCHS_TRIAL, trial, fold, log_dir
        )
        
        fold_scores.append(best_val_loss)
    
    mean_score = np.mean(fold_scores)
    print(f"  Trial {trial.number} completed: {mean_score:.4f} ± {np.std(fold_scores):.4f}")
    
    return mean_score

def main():
    """Main function"""
    
    print("Starting Optuna Hyperparameter Optimization")
    print("=" * 50)
    
    # Create study
    study_name = f"architecture_search_{timestamp}"
    storage = f"sqlite:///{join(OPTUNA_DIR, study_name)}.db"
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETED")
    print("=" * 50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    
    with open(join(OPTUNA_DIR, f'results_{study_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Train final models
    print("\n" + "=" * 50)
    print("TRAINING FINAL MODELS")
    print("=" * 50)
    
    # Get best configurations
    best_trial = study.best_trial
    arch_config = suggest_architecture(best_trial)
    train_config = suggest_training_params(best_trial)
    
    # Train final models for each fold
    for fold in range(N_FOLDS):
        print(f'Training final model - Fold {fold+1}/{N_FOLDS}')
        
        # Create model
        model = ConfigurableModel(arch_config).to(device)
        
        # Prepare training and validation splits
        full_dataset = Dataset(DATASET_DIR, DATA_TYPE, 'train', fold)
        val_ratio = 0.2
        train_size = int(len(full_dataset) * (1 - val_ratio))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        test_dataset = Dataset(DATASET_DIR, DATA_TYPE, 'test', fold)

        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
        
        # Create optimizer and loss
        criterion = create_loss_function(train_config['loss_function'])
        optimizer = create_optimizer(model, train_config['optimizer'], 
                                   train_config['learning_rate'], train_config['weight_decay'])
        
        # Create scheduler
        scheduler = None
        if train_config['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=train_config['scheduler_factor'],
                patience=train_config['scheduler_patience'], min_lr=1e-7
            )
        
        # Early stopping (use longer patience for final training)
        early_stopping = EarlyStopping(
            patience=FINAL_EARLY_STOPPING_PATIENCE,  # Use original 110 epochs patience
            min_delta=0.001
        )
        early_stopping.max_grad_norm = train_config['max_grad_norm']
        
        # Load scaler
        scaler = load(open(join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl"), 'rb'))
        
        # Setup logging
        log_dir = join(LOGS_DIR, 'final_models', f'fold_{fold}')
        ensure_dir(log_dir)
        
        # Train final model using validation set for early stopping
        best_val_loss, _ = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            early_stopping, scaler, MAX_EPOCHS_FINAL, None, fold, log_dir, is_final=True
        )

        # Evaluate best model on train and test sets
        train_loss, train_x_nrmse, train_y_nrmse, train_z_nrmse = evaluate(model, train_loader, criterion, scaler)
        test_loss, test_x_nrmse, test_y_nrmse, test_z_nrmse = evaluate(model, test_loader, criterion, scaler)

        # Log final metrics
        writer_train = SummaryWriter(join(log_dir, 'train'))
        writer_test = SummaryWriter(join(log_dir, 'test'))
        writer_train.add_hparams(
            {
                'sess': 'train',
                'Type': DATA_TYPE,
                'lr': train_config['learning_rate'],
                'bsize': train_config['batch_size'],
                'DS': DATASET_NAME,
                'lossFunc': train_config['loss_function'],
            },
            {
                'loss': train_loss,
                'X_nRMSE': train_x_nrmse,
                'Y_nRMSE': train_y_nrmse,
                'Z_nRMSE': train_z_nrmse,
            },
        )
        writer_test.add_hparams(
            {
                'sess': 'test',
                'Type': DATA_TYPE,
                'lr': train_config['learning_rate'],
                'bsize': train_config['batch_size'],
                'DS': DATASET_NAME,
                'lossFunc': train_config['loss_function'],
            },
            {
                'loss': test_loss,
                'X_nRMSE': test_x_nrmse,
                'Y_nRMSE': test_y_nrmse,
                'Z_nRMSE': test_z_nrmse,
            },
        )
        writer_train.close()
        writer_test.close()
        
        # Save model
        model_dir = join(MODELS_DIR, 'final_models')
        ensure_dir(model_dir)
        
        model_path = join(model_dir, f'{DATA_TYPE}_fold_{fold}_optuna_best.pt')
        torch.jit.script(model).save(model_path)

        print(f'  Final model saved: {model_path}')
        print(f'  Best validation loss: {best_val_loss:.4f}')
        print(f'  Test loss: {test_loss:.4f} (X:{test_x_nrmse:.4f}, Y:{test_y_nrmse:.4f}, Z:{test_z_nrmse:.4f})')
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Study database: {storage}")
    print(f"Results saved to: {join(OPTUNA_DIR, f'results_{study_name}.json')}")
    print(f"Final models saved to: {join(MODELS_DIR, 'final_models')}")
    print(f"TensorBoard logs: {LOGS_DIR}")
    print("\nTo view results:")
    print(f"  optuna-dashboard {storage}")
    print(f"  tensorboard --logdir {LOGS_DIR}")

if __name__ == "__main__":
    main()
