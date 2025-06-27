import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
from pickle import load
from tqdm import tqdm
import random
import datetime

######### Configuration Section ########
# Experiment and model naming
exp_name = 'date_Dense_1st_torch'  # Experiment name for logging
modelVersion = 'Dense_1st_torch'    # Version identifier for the model
nameDataset = 'IWALQQ_1st_correction'  # Name of the dataset being used
dataType = 'angle'  # Type of data to predict: 'angle' or 'moBWHT'

# Training hyperparameters
learningRate = 0.0005  # Learning rate for the optimizer
batch_size = 64        # Number of samples per batch
lossFunction = "RMSE"  # Loss function to use: "RMSE" or "MAE"

# Cross-validation and training settings
totalFold = 5     # Number of folds for cross-validation
epochs = 1000     # Number of training epochs per fold

log_interval = 10  # Interval for logging (currently not used in main loop)

# Directory paths for data and model storage
# Data directory containing the preprocessed datasets
absDataDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"

dataSetDir = join(absDataDir, nameDataset)  # Full path to dataset
# Model saving directory
SaveDir = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\trainedModel'
# TensorBoard logs directory
logDir = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\training\logs'
############################

print(f"Current settings - Type:{dataType}, lr:{learningRate}, BS:{batch_size}, LF:{lossFunction},\
     \nmodelV:{modelVersion}, DataSet:{nameDataset}")

# Generate timestamp for unique identification
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-2]

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Multi-Layer Perceptron (MLP) model
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        # Flatten layer to convert input to 1D
        self.flatten = nn.Flatten()
        # First fully connected layer: input size 4242 -> 6000 neurons
        self.layer1 = nn.Linear(4242, 6000)
        # Second fully connected layer: 6000 -> 4000 neurons
        self.layer2 = nn.Linear(6000, 4000)
        # Output layer: 4000 -> 303 neurons (final prediction size)
        self.layer3 = nn.Linear(4000, 303)
        # Dropout layer for regularization (50% dropout rate)
        self.dropout1 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Flatten the input tensor
        x = self.flatten(x)
        # First layer with ReLU activation
        x = F.relu(self.layer1(x))
        # Apply dropout for regularization
        x = self.dropout1(x)
        # Second layer with ReLU activation
        x = F.relu(self.layer2(x))
        # Output layer (no activation - regression task)
        x = self.layer3(x)
        return x

# Custom RMSE (Root Mean Square Error) loss function
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()  # Use MSE as base
        self.eps = eps  # Small epsilon to prevent sqrt(0)
        
    def forward(self, yhat, y):
        # Calculate RMSE: sqrt(MSE + epsilon)
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

# Factory function to create loss functions based on string identifier
def makelossFuncion(lossFunction):
    """
    Creates and returns the appropriate loss function
    Args:
        lossFunction (str): Either "RMSE" or "MAE"
    Returns:
        loss function object
    """
    if lossFunction == "RMSE":
        criterion = RMSELoss()  # Custom RMSE loss
    elif lossFunction == 'MAE':
        criterion = nn.L1Loss()  # Mean Absolute Error
    return criterion

# Custom Dataset class for loading IMU data
class Dataset(torch.utils.data.Dataset): 
    def __init__(self, dataSetDir, dataType, sess, numFold):
        """
        Initialize dataset loader
        Args:
            dataSetDir (str): Directory containing the dataset files
            dataType (str): Type of data ('angle' or 'moBWHT')
            sess (str): Session type ('train' or 'test')
            numFold (int): Current fold number for cross-validation
        """
        self.dataType = dataType  # Store data type
        self.sess = sess  # Store session type
        
        # Load input features (X) from numpy archive
        self.load_Data_X = torch.from_numpy(
            np.load(join(dataSetDir, f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_X_{self.sess}']
        )
        # Load target labels (Y) from numpy archive
        self.load_Data_Y = torch.from_numpy(
            np.load(join(dataSetDir, f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_Y_{self.dataType}_{self.sess}']
        )
        
        # Convert to float tensors and remove unnecessary dimensions
        self.load_Data_X = torch.squeeze(self.load_Data_X.type(torch.FloatTensor))
        self.load_Data_Y = torch.squeeze(self.load_Data_Y.type(torch.FloatTensor))
        
    def __len__(self):
        """Return the total number of samples"""
        return len(self.load_Data_X)

    def __getitem__(self, idx):
        """
        Get a single sample by index
        Args:
            idx (int): Index of the sample
        Returns:
            tuple: (input_features, target_labels)
        """
        X = self.load_Data_X[idx]
        Y = self.load_Data_Y[idx]
        return X, Y

def ensure_dir(file_path):
    """
    Create directory if it doesn't exist
    Args:
        file_path (str): Path to directory
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def nRMSE_Axis_TLPerbatch(pred, target, axis, load_scaler4Y):
    """
    Calculate normalized RMSE for a specific axis across a batch
    
    Args:
        pred (torch.Tensor): Predicted values
        target (torch.Tensor): Ground truth values
        axis (str): Axis to calculate nRMSE for ('x', 'y', or 'z')
        load_scaler4Y: Scaler object for denormalization
    
    Returns:
        float: Total nRMSE across the batch for the specified axis
    """
    # Map axis strings to indices
    dict_axis = {
        'x': 0,
        "y": 1,
        "z": 2,
    }
    axis = dict_axis[axis]
    nRMSE_perbatch = 0
    
    batchNum = len(target)  # Get batch size
    
    # Iterate through each sample in the batch
    for bat in range(batchNum):
        # Reshape predictions and targets to separate x, y, z components
        # Convert from flat vector to [timesteps, 3] format and extract specific axis
        pred_axis = torch.transpose(torch.reshape(torch.squeeze(pred[bat]), [3, -1]), 0, 1)[:, axis]
        target_axis = torch.transpose(torch.reshape(torch.squeeze(target[bat]), [3, -1]), 0, 1)[:, axis]
        
        # Denormalize the data using the scaler
        pred_axis = (pred_axis - torch.tensor(load_scaler4Y.min_[axis])) / torch.tensor(load_scaler4Y.scale_[axis])
        target_axis = (target_axis - torch.tensor(load_scaler4Y.min_[axis])) / torch.tensor(load_scaler4Y.scale_[axis])
        
        # Calculate normalized RMSE: 100 * RMSE / (max - min)
        nRMSE = 100 * torch.sqrt(torch.mean(torch.square(pred_axis - target_axis))) / (torch.max(target_axis) - torch.min(target_axis))
        nRMSE_perbatch += nRMSE
        
    return nRMSE_perbatch

# Main training loop - iterate through each fold of cross-validation
for numFold in range(totalFold):
    print(f'Current fold: {numFold + 1}/{totalFold}')
    
    # Create a new model instance for each fold
    my_model = Mlp()
    my_model.to(device)  # Move model to GPU/CPU
    
    # Initialize loss function and optimizer
    criterion = makelossFuncion(lossFunction)
    optimizer = torch.optim.NAdam(my_model.parameters(), lr=learningRate)

    # Create dataset instances for training and testing
    angle_train = Dataset(dataSetDir, dataType, 'train', numFold)
    angle_test = Dataset(dataSetDir, dataType, 'test', numFold)
    
    # Create data loaders for batch processing
    train_loader = DataLoader(angle_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(angle_test, batch_size=batch_size, shuffle=True)

    # Initialize TensorBoard writers for logging training metrics
    writer_train = SummaryWriter(join(logDir, f'{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_LF_{lossFunction}/train/{numFold}_fold'))
    writer_test = SummaryWriter(join(logDir, f'{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_LF_{lossFunction}/test/{numFold}_fold'))
    
    # Add model graph to TensorBoard (for visualization)
    x = torch.rand(1, 4242, device=device)  # Create dummy input
    writer_train.add_graph(my_model, x)
    writer_test.add_graph(my_model, x)

    # Load the scaler used for denormalization in metric calculation
    load_scaler4Y = load(open(join(dataSetDir, f"{numFold}_fold_scaler4Y_{dataType}.pkl"), 'rb'))
    
    # Training loop for each epoch
    for epoch in range(epochs):
        # Set model to training mode (enables dropout, batch norm training mode)
        my_model.train()
        
        # Initialize training metrics
        train_loss = 0
        train_x_nRMSE = 0
        train_y_nRMSE = 0
        train_z_nRMSE = 0
        
        # Training phase - iterate through training batches
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move data to device (GPU/CPU)
            data, target = data.to(device), target.to(device)
            
            # Clear gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            output = my_model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Accumulate training loss (weighted by batch size for proper averaging)
            train_loss += loss.item() * data.size(0)
            
            # Calculate and accumulate nRMSE for each axis
            train_x_nRMSE += nRMSE_Axis_TLPerbatch(output, target, 'x', load_scaler4Y).item()
            train_y_nRMSE += nRMSE_Axis_TLPerbatch(output, target, 'y', load_scaler4Y).item()
            train_z_nRMSE += nRMSE_Axis_TLPerbatch(output, target, 'z', load_scaler4Y).item()

        # Calculate average training metrics across all samples
        train_loss /= len(train_loader.sampler)
        train_x_nRMSE /= len(train_loader.sampler)
        train_y_nRMSE /= len(train_loader.sampler)
        train_z_nRMSE /= len(train_loader.sampler)
        
        # Log training metrics to TensorBoard
        writer_train.add_scalar('loss(MAE)', train_loss, epoch)
        writer_train.add_scalar(f'{dataType}_X_nRMSE', train_x_nRMSE, epoch)
        writer_train.add_scalar(f'{dataType}_Y_nRMSE', train_y_nRMSE, epoch)
        writer_train.add_scalar(f'{dataType}_Z_nRMSE', train_z_nRMSE, epoch)

        # Evaluation phase
        test_loss = 0
        test_x_nRMSE = 0
        test_y_nRMSE = 0
        test_z_nRMSE = 0
        
        # Set model to evaluation mode (disables dropout, batch norm eval mode)
        my_model.eval()
        
        # Disable gradient computation for evaluation (saves memory and speeds up)
        with torch.no_grad():
            # Iterate through test batches
            for data, target in test_loader:
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Forward pass: compute predictions
                output = my_model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                
                # Accumulate test metrics
                test_loss += loss.item() * data.size(0)
                test_x_nRMSE += nRMSE_Axis_TLPerbatch(output, target, 'x', load_scaler4Y).item()
                test_y_nRMSE += nRMSE_Axis_TLPerbatch(output, target, 'y', load_scaler4Y).item()
                test_z_nRMSE += nRMSE_Axis_TLPerbatch(output, target, 'z', load_scaler4Y).item()

            # Calculate average test metrics
            test_loss /= len(test_loader.sampler)
            test_x_nRMSE /= len(test_loader.sampler)
            test_y_nRMSE /= len(test_loader.sampler)
            test_z_nRMSE /= len(test_loader.sampler)

            # Log test metrics to TensorBoard
            writer_test.add_scalar('loss(MAE)', test_loss, epoch)
            writer_test.add_scalar(f'{dataType}_X_nRMSE', test_x_nRMSE, epoch)
            writer_test.add_scalar(f'{dataType}_Y_nRMSE', test_y_nRMSE, epoch)
            writer_test.add_scalar(f'{dataType}_Z_nRMSE', test_z_nRMSE, epoch)

        # Print epoch results
        print(f'\nTrain set: Average loss: {train_loss:.4f}, X_nRMSE: {train_x_nRMSE}, Y_nRMSE: {train_y_nRMSE}, Z_nRMSE: {train_z_nRMSE}'
             + f'\nTest set: Average loss: {test_loss:.4f}, X_nRMSE: {test_x_nRMSE}, Y_nRMSE: {test_y_nRMSE}, Z_nRMSE: {test_z_nRMSE}')
    
    # Log hyperparameters and final metrics to TensorBoard
    writer_train.add_hparams(
        {"sess": "train", "Type": dataType, "lr": learningRate, "bsize": batch_size, "DS": nameDataset, 'lossFunc': lossFunction},
        {
            "loss": train_loss,
            'X_nRMSE': train_x_nRMSE,
            'Y_nRMSE': train_y_nRMSE,
            'Z_nRMSE': train_z_nRMSE,
        },
    )
    writer_test.add_hparams(
        {"sess": "test", "Type": dataType, "lr": learningRate, "bsize": batch_size, "DS": nameDataset, 'lossFunc': lossFunction},
        {
            "loss": test_loss,
            'X_nRMSE': test_x_nRMSE,
            'Y_nRMSE': test_y_nRMSE,
            'Z_nRMSE': test_z_nRMSE,
        },
    )
    
    # Close TensorBoard writers
    writer_train.close()
    writer_test.close()
    
    # Save the trained model
    dir_save_torch = join(SaveDir, modelVersion, nameDataset)
    ensure_dir(dir_save_torch)  # Create directory if it doesn't exist
    
    # Convert model to TorchScript format for deployment
    model_scripted = torch.jit.script(my_model)
    model_scripted.save(join(dir_save_torch, f'{dataType}_{numFold}_fold.pt'))
    
    print(f"Model saved for fold {numFold}")

print("Training completed for all folds!")

# Notes for loading saved models:
# Always call model.eval() after loading a model for inference!
# Example:
# model = torch.jit.load('model_scripted.pt')
# model.eval()
# See: https://tutorials.pytorch.kr/beginner/saving_loading_models.html#export-load-model-in-torchscript-format

