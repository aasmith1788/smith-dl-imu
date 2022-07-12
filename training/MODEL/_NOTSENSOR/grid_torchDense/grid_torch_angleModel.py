import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
from pickle import load
from tqdm import tqdm
import random
import datetime
from CBDtorch.custom.metric_NOTSENSOR import nRMSE_Axis_TLPerbatch

######### 설정 영역 ########
exp_name = "NOTSENSOR_DenseModel"  # 실험 이름 혹은 오늘 날짜
modelVersion = "Dense_1st_torch"
nameDataset = "IWALQQ_AE_4th_NOTSENSOR"
dataType = "angle"  # or moBWHT

#################################
# 여기는 grid로 돌림!
#################################
list_learningRate = {0: 0.006, 1: 0.008, 2: 0.01}  # opt1 {0:0.006, 1:0.008, 2:0.01}
list_batch_size = {0: 128}  # opt2
list_lossFunction = {0: "MAE"}  # opt2

totalFold = 5
epochs = 1000

log_interval = 10  # 모델 저장 위치
# 저장위치
# 데이터 위치
absDataDir = r"/restricted/project/movelab/bcha/IMUforKnee/preperation/SAVE_dataSet"
dataSetDir = join(absDataDir, nameDataset)
# 모델 위치
SaveDir = r"/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel"
logDir = r"/restricted/project/movelab/bcha/IMUforKnee/training/logs"
############################

# CPU or GPU?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 만들기
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(4242, 6000)
        self.layer2 = nn.Linear(6000, 4000)
        self.layer3 = nn.Linear(4000, 303)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# 검사할 lossfunction 생기면 여기다가 추가할 것
def makelossFuncion(lossFunction):
    if lossFunction == "RMSE":
        criterion = RMSELoss()  # mean absolute error
    elif lossFunction == "MAE":
        criterion = nn.L1Loss()
    return criterion


# 개빠른가..?
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataSetDir, dataType, sess, numFold):
        self.dataType = dataType  # angle or moBWHT
        self.sess = sess  # train or test
        self.load_Data_X = torch.from_numpy(
            np.load(join(dataSetDir, f"{str(numFold)}_fold_final_{sess}.npz"))[
                f"final_X_{self.sess}"
            ]
        )
        self.load_Data_Y = torch.from_numpy(
            np.load(join(dataSetDir, f"{str(numFold)}_fold_final_{sess}.npz"))[
                f"final_Y_{self.dataType}_{self.sess}"
            ]
        )
        # AE데이터로 Dense 학습 시킬 때
        self.load_Data_X = np.reshape(
            self.load_Data_X,
            newshape=(-1, self.load_Data_X.shape[1] * self.load_Data_X.shape[2], 1),
            order="F",
        )
        self.load_Data_Y = np.reshape(
            self.load_Data_Y,
            newshape=(-1, self.load_Data_Y.shape[1] * self.load_Data_Y.shape[2], 1),
            order="F",
        )

        self.load_Data_X = torch.squeeze(self.load_Data_X.type(torch.FloatTensor))
        self.load_Data_Y = torch.squeeze(self.load_Data_Y.type(torch.FloatTensor))

    def __len__(self):
        return len(self.load_Data_X)

    def __getitem__(self, idx):
        X = self.load_Data_X[idx]
        Y = self.load_Data_Y[idx]
        return X, Y


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


# batch당 총 error를 누적해서 줌
# TL = Total Loss per batch


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


class MinMaxScalerSensor(MinMaxScaler):
    def fit(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], -1))
        super().fit(x, y=y)

    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], -1))
        return np.reshape(super().transform(x), newshape=X.shape)

    def inverse_transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], -1))
        return np.reshape(super().inverse_transform(x), newshape=X.shape)


count = 0
for opt1 in range(0, len(list_learningRate)):
    for opt2 in range(0, len(list_batch_size)):
        learningRate = list_learningRate[opt1]
        batch_size = list_batch_size[opt2]
        lossFunction = list_lossFunction[0]
        print(
            f"count:{count} | 현재 설정 Type:{dataType}, lr:{learningRate}, BS:{batch_size}, LF:{lossFunction},\
            \nmodelV:{modelVersion}, DataSet:{nameDataset}"
        )
        count = count + 1
        # 시간 설정
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-2]
        for numFold in range(totalFold):
            print(f"now fold: {numFold}")
            # 매 fold마다 새로운 모델
            my_model = Mlp()
            my_model.to(device)

            # loss function and optimizer define
            criterion = makelossFuncion(lossFunction)
            optimizer = torch.optim.NAdam(my_model.parameters(), lr=learningRate)

            angle_train = Dataset(dataSetDir, dataType, "train", numFold)
            angle_test = Dataset(dataSetDir, dataType, "test", numFold)
            train_loader = DataLoader(angle_train, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(angle_test, batch_size=batch_size, shuffle=True)

            # 시각화를 위한 tensorboard 초기화
            writer_train = SummaryWriter(
                join(
                    logDir,
                    f"{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_LF_{lossFunction}/train/{numFold}_fold",
                )
            )
            writer_test = SummaryWriter(
                join(
                    logDir,
                    f"{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_LF_{lossFunction}/test/{numFold}_fold",
                )
            )
            x = torch.rand(1, 4242, device=device)
            writer_train.add_graph(my_model, x)
            writer_test.add_graph(my_model, x)

            # 학습시작 전 metric용 scaler 불러오기
            load_scaler4Y = load(
                open(join(dataSetDir, f"{numFold}_fold_scaler4Y_{dataType}.pkl"), "rb")
            )
            for epoch in range(epochs):
                my_model.train()

                train_loss = 0
                train_x_nRMSE = 0
                train_y_nRMSE = 0
                train_z_nRMSE = 0
                for batch_idx, (data, target) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                ):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = my_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(
                        0
                    )  # 이것은 모든 배치의 크기가 일정하지 않을 수 있기 때문에 이렇게 수행함! train_loss는 total loss of batch가 됨

                    train_x_nRMSE += nRMSE_Axis_TLPerbatch(
                        output, target, "x", load_scaler4Y, device
                    ).item()  # 해당 배치에서의 총 loss의 합
                    train_y_nRMSE += nRMSE_Axis_TLPerbatch(
                        output, target, "y", load_scaler4Y, device
                    ).item()  # 해당 배치에서의 총 loss의 합
                    train_z_nRMSE += nRMSE_Axis_TLPerbatch(
                        output, target, "z", load_scaler4Y, device
                    ).item()  # 해당 배치에서의 총 loss의 합
                    # if batch_idx % log_interval == 0:
                    #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #         epoch, batch_idx * len(data), len(train_loader.dataset),
                    #         100. * batch_idx / len(train_loader), loss.item()))

                train_loss /= len(train_loader.sampler)
                train_x_nRMSE /= len(train_loader.sampler)
                train_y_nRMSE /= len(train_loader.sampler)
                train_z_nRMSE /= len(train_loader.sampler)

                writer_train.add_scalar("loss(MAE)", train_loss, epoch)
                writer_train.add_scalar(f"{dataType}_X_nRMSE", train_x_nRMSE, epoch)
                writer_train.add_scalar(f"{dataType}_Y_nRMSE", train_y_nRMSE, epoch)
                writer_train.add_scalar(f"{dataType}_Z_nRMSE", train_z_nRMSE, epoch)

                test_loss = 0
                test_x_nRMSE = 0
                test_y_nRMSE = 0
                test_z_nRMSE = 0
                my_model.eval()  # batch norm이나 dropout 등을 train mode 변환
                with torch.no_grad():  # autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = my_model(data)
                        loss = criterion(output, target)
                        test_loss += loss.item() * data.size(0)
                        test_x_nRMSE += nRMSE_Axis_TLPerbatch(
                            output, target, "x", load_scaler4Y, device
                        ).item()  # 해당 배치에서의 총 loss의 합
                        test_y_nRMSE += nRMSE_Axis_TLPerbatch(
                            output, target, "y", load_scaler4Y, device
                        ).item()  # 해당 배치에서의 총 loss의 합
                        test_z_nRMSE += nRMSE_Axis_TLPerbatch(
                            output, target, "z", load_scaler4Y, device
                        ).item()  # 해당 배치에서의 총 loss의 합

                    test_loss /= len(test_loader.sampler)
                    test_x_nRMSE /= len(test_loader.sampler)
                    test_y_nRMSE /= len(test_loader.sampler)
                    test_z_nRMSE /= len(test_loader.sampler)

                    writer_test.add_scalar("loss(MAE)", test_loss, epoch)
                    writer_test.add_scalar(f"{dataType}_X_nRMSE", test_x_nRMSE, epoch)
                    writer_test.add_scalar(f"{dataType}_Y_nRMSE", test_y_nRMSE, epoch)
                    writer_test.add_scalar(f"{dataType}_Z_nRMSE", test_z_nRMSE, epoch)

                print(
                    f"\nTrain set: Average loss: {train_loss:.4f}, X_nRMSE: {train_x_nRMSE}, Y_nRMSE: {train_y_nRMSE}, Z_nRMSE: {train_z_nRMSE}"
                    + f"\nTest set: Average loss: {test_loss:.4f}, X_nRMSE: {test_x_nRMSE}, Y_nRMSE: {test_y_nRMSE}, Z_nRMSE: {test_z_nRMSE}"
                )
            writer_train.add_hparams(
                {
                    "sess": "train",
                    "Type": dataType,
                    "lr": learningRate,
                    "bsize": batch_size,
                    "DS": nameDataset,
                    "lossFunc": lossFunction,
                },
                {
                    "loss": train_loss,
                    "X_nRMSE": train_x_nRMSE,
                    "Y_nRMSE": train_y_nRMSE,
                    "Z_nRMSE": train_z_nRMSE,
                },
            )
            writer_test.add_hparams(
                {
                    "sess": "test",
                    "Type": dataType,
                    "lr": learningRate,
                    "bsize": batch_size,
                    "DS": nameDataset,
                    "lossFunc": lossFunction,
                },
                {
                    "loss": test_loss,
                    "X_nRMSE": test_x_nRMSE,
                    "Y_nRMSE": test_y_nRMSE,
                    "Z_nRMSE": test_z_nRMSE,
                },
            )
            writer_train.close()
            writer_test.close()
            dir_save_torch = join(SaveDir, modelVersion, nameDataset)
            ensure_dir(dir_save_torch)
            model_scripted = torch.jit.script(my_model)  # Export to TorchScript
            model_scripted.save(
                join(dir_save_torch, f"{dataType}_{numFold}_fold.pt")
            )  # Save
            # 저장된 모델 불러올 때
            # 항상 불러온 모델 뒤에 model.eval() 붙일 것!
            # https://tutorials.pytorch.kr/beginner/saving_loading_models.html#export-load-model-in-torchscript-format
            # model = torch.jit.load('model_scripted.pt')
            # model.eval()
