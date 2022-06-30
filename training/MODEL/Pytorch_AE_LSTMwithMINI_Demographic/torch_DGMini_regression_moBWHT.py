# 필요한 라이브러리 설정
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.testing import make_tensor
from torch.utils.tensorboard import SummaryWriter

from os.path import join
from pickle import load
from tqdm import tqdm
import datetime

from CBDtorch.dense_dg_mini import *
from CBDtorch.custom import DatasetWithDG4regressor
from CBDtorch.custom import MinMaxScalerSensor
from CBDtorch.dirs import *
from CBDtorch.custom.metric import nRMSE_Axis_TLPerbatch

######### 설정 영역 ########
# 실험 관련 세팅
exp_name = "DGMini_3rd4th"  # # 실험 이름 혹은 오늘 날짜
modelVersion = "DG_DenseRegressor_1st_torch_mini" #weight_decay를 표시하기
# 이모델에서 사용할 vaelstm 모델 이름
vae_ModelVersion = "vaelstm_3rd_torch"
nameDataset = "IWALQQ_AE_4th"
load_dataType = "angle"
dataType = "moBWHT"  # moBWHT
# 데이터 feature 정보, 추후에 자동화가 필요할랑가?
seq_len = 101
num_features = 42
#################################
# 여기는 grid로 돌림! 이제 grid 포함이 default!
#################################
list_embedding_dim = [5, 10, 20, 30, 40, 50, 60, 70, 80]
list_learningRate = [0.001]  # opt1
list_batch_size = {0: 128}  # opt2
list_lossFunction = {0: "MAE"}  # opt2
weight_decay = 0.0 # 0.001 or 0.0005 둘 중 하나

totalFold = 5  # total fold, I did 5-fold cross validation
epochs = 500  # total epoch
log_interval = 10  # frequency for saving log file

# 데이터 위치
absDataDir = r"/restricted/project/movelab/bcha/IMUforKnee/preperation/SAVE_dataSet"
dataSetDir = join(absDataDir, nameDataset)
# 모델 위치
SaveDir = r"/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel"
logDir = r"/restricted/project/movelab/bcha/IMUforKnee/training/logs"
############################
# 필요한 함수 설정

############################
# cuda or cpu?
device = "cuda" if torch.cuda.is_available() else "cpu"

# grid search 시, 현재의 count
count = 0
for opt1 in range(0, len(list_learningRate)):
    for opt2 in range(0, len(list_lossFunction)):
        for opt3 in range(0, len(list_embedding_dim)):
            for opt4 in range(0, len(list_batch_size)):
                learningRate = list_learningRate[opt1]
                lossFunction = list_lossFunction[opt2]
                embedding_dim = list_embedding_dim[opt3]
                batch_size = list_batch_size[opt4]
                print(
                    f"count:{count} | 현재 설정 Type:{dataType}, lr:{learningRate}, BS:{batch_size}, LF:{lossFunction}, emb_dim:{embedding_dim}, \
                    \nmodelV:{modelVersion}, DataSet:{nameDataset}"
                )
                count = count + 1
                # 조건 별로 시작 시간 기록하기
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-2]
                for numFold in range(totalFold):
                    print(f"now fold: {numFold}")
                    # 매 fold마다 새로운 모델
                    # 학습된 VAE 불러오기
                    dir_savedVAE = join(SaveDir, vae_ModelVersion, nameDataset)
                    loadmodelname = join(
                        dir_savedVAE, f"{load_dataType}_{embedding_dim}_{numFold}_fold"
                    )
                    my_model = regressor(
                        loadmodelname,
                        embedding_dim,
                        seq_len,
                        num_features,
                        embedding_dim,
                        device,
                    )
                    my_model.to(device)

                    # loss function and optimizer define
                    optimizer = torch.optim.NAdam(
                        my_model.dense.parameters(),
                        lr=learningRate,
                        weight_decay=weight_decay,
                    )

                    data_train = DatasetWithDG4regressor(
                        dataSetDir, dataType, "train", numFold
                    )
                    data_test = DatasetWithDG4regressor(
                        dataSetDir, dataType, "test", numFold
                    )
                    train_loader = DataLoader(
                        data_train, batch_size=batch_size, shuffle=True
                    )
                    test_loader = DataLoader(
                        data_test, batch_size=batch_size, shuffle=True
                    )

                    # 시각화를 위한 tensorboard 초기화
                    writer_train = SummaryWriter(
                        join(
                            logDir,
                            f"{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_embdim_{embedding_dim}_weightDecay_{weight_decay}/train/{numFold}_fold",
                        )
                    )
                    writer_test = SummaryWriter(
                        join(
                            logDir,
                            f"{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_embdim_{embedding_dim}_weightDecay_{weight_decay}/test/{numFold}_fold",
                        )
                    )

                    load_scaler4Y = load(
                        open(
                            join(dataSetDir, f"{numFold}_fold_scaler4Y_{dataType}.pkl"),
                            "rb",
                        )
                    )
                    # 데이터 그래프 기록
                    my_model.eval()
                    _, (data_imu, data_dg, target) = next(enumerate(tqdm(train_loader)))
                    data_imu, data_dg, target = (
                        data_imu.to(device),
                        data_dg.to(device),
                        target.to(device),
                    )
                    writer_train.add_graph(my_model, (data_imu, data_dg))
                    writer_test.add_graph(my_model, (data_imu, data_dg))

                    for epoch in range(epochs):
                        # train session
                        my_model.train()
                        train_loss = 0
                        train_x_nRMSE = 0
                        train_y_nRMSE = 0
                        train_z_nRMSE = 0
                        for batch_idx, (data_imu, data_dg, target) in enumerate(
                            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                        ):
                            # print(f'datainput shape:{data.shape}')
                            data_imu, data_dg, target = (
                                data_imu.to(device),
                                data_dg.to(device),
                                target.to(device),
                            )
                            optimizer.zero_grad()
                            output = my_model(data_imu, data_dg)
                            criterion = nn.L1Loss()
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * data_imu.size(
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

                        train_loss /= len(train_loader.sampler)
                        train_x_nRMSE /= len(train_loader.sampler)
                        train_y_nRMSE /= len(train_loader.sampler)
                        train_z_nRMSE /= len(train_loader.sampler)

                        writer_train.add_scalar("loss(MAE)", train_loss, epoch)
                        writer_train.add_scalar(
                            f"{dataType}_X_nRMSE", train_x_nRMSE, epoch
                        )
                        writer_train.add_scalar(
                            f"{dataType}_Y_nRMSE", train_y_nRMSE, epoch
                        )
                        writer_train.add_scalar(
                            f"{dataType}_Z_nRMSE", train_z_nRMSE, epoch
                        )

                        # test session
                        test_loss = 0
                        test_x_nRMSE = 0
                        test_y_nRMSE = 0
                        test_z_nRMSE = 0
                        my_model.eval()  # batch norm이나 dropout 등을 train mode 변환
                        with torch.no_grad():  # autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임
                            for data_imu, data_dg, target in test_loader:
                                data_imu, data_dg, target = (
                                    data_imu.to(device),
                                    data_dg.to(device),
                                    target.to(device),
                                )
                                output = my_model(data_imu, data_dg)
                                loss = criterion(output, target)
                                test_loss += loss.item() * data_imu.size(0)
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
                            writer_test.add_scalar(
                                f"{dataType}_X_nRMSE", test_x_nRMSE, epoch
                            )
                            writer_test.add_scalar(
                                f"{dataType}_Y_nRMSE", test_y_nRMSE, epoch
                            )
                            writer_test.add_scalar(
                                f"{dataType}_Z_nRMSE", test_z_nRMSE, epoch
                            )

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
                            "emb_dim": embedding_dim,
                            "weight_Decay": weight_decay,
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
                            "emb_dim": embedding_dim,
                            "weight_Decay": weight_decay,
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
                    # 현재 아래 방법은 모델 저장이 안되는 단점이 있음
                    # model_scripted = torch.jit.script(my_model) # Export to TorchScript
                    # model_scripted.save(join(dir_save_torch,f'{dataType}_{numFold}_fold.pt')) # Save

                    # 수정된 방법
                    torch.save(
                        my_model.state_dict(),
                        join(
                            dir_save_torch, f"{dataType}_{embedding_dim}_{numFold}_fold"
                        ),
                    )  # 모델 정의가 필요함
                    # 수정된 방법으로 모델 불러올 떄
                    # Model class must be defined somewhere
                    # model.load_state_dict(torch.load(filepath))
                    # model.eval()
