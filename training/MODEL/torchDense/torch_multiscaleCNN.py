# Multi-scale 1D CNN training script for IMU joint angle prediction

import os
from os.path import join
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pickle import load


class CNNDataset(torch.utils.data.Dataset):
    """Dataset that reshapes flattened IMU sequences to (42, 101)."""

    def __init__(self, dataSetDir, dataType, sess, fold):
        self.dataType = dataType
        self.sess = sess
        data = np.load(join(dataSetDir, f"{fold}_fold_final_{sess}.npz"))
        X = np.squeeze(data[f"final_X_{sess}"]).astype(np.float32)
        Y = np.squeeze(data[f"final_Y_{dataType}_{sess}"]).astype(np.float32)
        # reshape from (4242,) -> (42, 101)
        X = X.reshape(-1, 42, 101)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MultiScaleCNN(nn.Module):
    """1D CNN with parallel branches of different kernel sizes."""

    def __init__(self, num_features=42, num_outputs=303, dropout=0.5):
        super().__init__()
        self.branch3 = self._make_branch(num_features, kernel=3)
        self.branch5 = self._make_branch(num_features, kernel=5)
        self.branch7 = self._make_branch(num_features, kernel=7)
        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

    @staticmethod
    def _make_branch(in_ch, kernel):
        padding = kernel // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=kernel, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=kernel, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=kernel, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        # x: (batch, 42, 101)
        b3 = self.branch3(x).squeeze(-1)
        b5 = self.branch5(x).squeeze(-1)
        b7 = self.branch7(x).squeeze(-1)
        out = torch.cat([b3, b5, b7], dim=1)
        out = self.fc(out)
        return out


def nRMSE_Axis_TLPerbatch(pred, target, axis, scaler):
    """Compute normalized RMSE per axis for a batch."""

    axis_dict = {"x": 0, "y": 1, "z": 2}
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    exp_name = "multiscale_cnn"
    modelVersion = "MultiScaleCNN"
    nameDataset = "IWALQQ_1st_correction"
    dataType = "angle"
    learningRate = 5e-4
    batch_size = 64
    epochs = 100
    totalFold = 5
    absDataDir = "./preperation/SAVE_dataSet"
    dataSetDir = join(absDataDir, nameDataset)
    output_base_dir = "./training_results/MultiScaleCNN"
    SaveDir = join(output_base_dir, "models")
    logDir = join(output_base_dir, "logs")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensure_dir(SaveDir)
    ensure_dir(logDir)

    for fold in range(totalFold):
        model = MultiScaleCNN().to(device)
        optimizer = torch.optim.NAdam(model.parameters(), lr=learningRate)
        criterion = nn.MSELoss()

        train_data = CNNDataset(dataSetDir, dataType, "train", fold)
        test_data = CNNDataset(dataSetDir, dataType, "test", fold)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        train_log_dir = join(logDir, f"fold_{fold}", "train")
        test_log_dir = join(logDir, f"fold_{fold}", "test")
        ensure_dir(train_log_dir)
        ensure_dir(test_log_dir)
        writer_train = SummaryWriter(train_log_dir)
        writer_test = SummaryWriter(test_log_dir)

        scaler = load(open(join(dataSetDir, f"{fold}_fold_scaler4Y_{dataType}.pkl"), "rb"))

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            x_nrmse = y_nrmse = z_nrmse = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.size(0)
                x_nrmse += nRMSE_Axis_TLPerbatch(out.detach(), y, "x", scaler)
                y_nrmse += nRMSE_Axis_TLPerbatch(out.detach(), y, "y", scaler)
                z_nrmse += nRMSE_Axis_TLPerbatch(out.detach(), y, "z", scaler)
            train_loss /= len(train_loader.dataset)
            x_nrmse /= len(train_loader.dataset)
            y_nrmse /= len(train_loader.dataset)
            z_nrmse /= len(train_loader.dataset)
            writer_train.add_scalar("loss", train_loss, epoch)
            writer_train.add_scalar("x_nRMSE", x_nrmse, epoch)
            writer_train.add_scalar("y_nRMSE", y_nrmse, epoch)
            writer_train.add_scalar("z_nRMSE", z_nrmse, epoch)

            model.eval()
            test_loss = 0
            tx = ty = tz = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    loss = criterion(out, y)
                    test_loss += loss.item() * X.size(0)
                    tx += nRMSE_Axis_TLPerbatch(out, y, "x", scaler)
                    ty += nRMSE_Axis_TLPerbatch(out, y, "y", scaler)
                    tz += nRMSE_Axis_TLPerbatch(out, y, "z", scaler)
            test_loss /= len(test_loader.dataset)
            tx /= len(test_loader.dataset)
            ty /= len(test_loader.dataset)
            tz /= len(test_loader.dataset)
            writer_test.add_scalar("loss", test_loss, epoch)
            writer_test.add_scalar("x_nRMSE", tx, epoch)
            writer_test.add_scalar("y_nRMSE", ty, epoch)
            writer_test.add_scalar("z_nRMSE", tz, epoch)

        writer_train.close()
        writer_test.close()
        save_path = join(SaveDir, f"{modelVersion}_{fold}.pt")
        torch.jit.script(model).save(save_path)


if __name__ == "__main__":
    main()
