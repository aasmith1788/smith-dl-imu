# 필요한 라이브러리 설정
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.testing import make_tensor
from torch.utils.tensorboard import SummaryWriter

from os.path import join
from tqdm import tqdm
import datetime

from CBDtorch.vaelstm_1layer import * #항상 layer 숫자 확인하고 기록하기
from CBDtorch.custom import Dataset4autoencoder
from CBDtorch.dirs import *

######### 설정 영역 ########
# 실험 관련 세팅
exp_name = 'tor_vaelstm_20220531'  # 실험 이름 혹은 오늘 날짜
modelVersion = 'vaelstm_2nd_torch'
nameDataset = 'IWALQQ_AE_1st'
dataType = 'angle' # VAE 나 AE 모델에서는 안중요하지만 추후 모델 predict일 때 편하게 하기 위해서 패킹을 이렇게 해둠

# 데이터 feature 정보, 추후에 자동화가 필요할랑가?
seq_len = 101
num_features = 42
#################################
# 여기는 grid로 돌림! 이제 grid 포함이 default!
#################################
list_embedding_dim = [10, 20, 30, 40, 50, 60, 70, 80] # {0:10, 0:20, 0:30, 1:40, 2:50, 3:60, 4:70} 
list_learningRate = {0: 0.0008}  # opt1
list_batch_size = {0: 128}  # opt2
list_lossFunction = {0: "VAE"}  # opt2

totalFold = 5  # total fold, I did 5-fold cross validation
epochs = 3000  # total epoch
log_interval = 10  # frequency for saving log file

# 데이터 위치
absDataDir = r'/restricted/project/movelab/bcha/IMUforKnee/preperation/SAVE_dataSet'
dataSetDir = join(absDataDir,nameDataset)
# 모델 위치
SaveDir = r'/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel'
logDir = r'/restricted/project/movelab/bcha/IMUforKnee/training/logs'
############################
# 필요한 함수 설정

############################
# cuda or cpu?
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# grid search 시, 현재의 count
count = 0
for opt1 in range(0,len(list_learningRate)):
    for opt2 in range(0,len(list_lossFunction)):
        for opt3 in range(0,len(list_embedding_dim)):
            for opt4 in range(0,len(list_batch_size)):
                learningRate = list_learningRate[opt1]
                lossFunction = list_lossFunction[opt2]
                embedding_dim = list_embedding_dim[opt3]
                batch_size = list_batch_size[opt4]                
                print(f"count:{count} | 현재 설정 Type:{dataType}, lr:{learningRate}, BS:{batch_size}, LF:{lossFunction}, emb_dim:{embedding_dim}, \
                    \nmodelV:{modelVersion}, DataSet:{nameDataset}")
                count = count + 1 
                # 조건 별로 시작 시간 기록하기
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-2]
                for numFold  in range(totalFold):
                    print(f'now fold: {numFold}')
                    # 매 fold마다 새로운 모델
                    my_model = RecurrentVariationalAutoencoder(seq_len, num_features, embedding_dim, device)
                    my_model.to(device)
                    
                    # loss function and optimizer define
                    optimizer = torch.optim.NAdam(my_model.parameters(),lr=learningRate)

                    angle_train = Dataset4autoencoder(dataSetDir, dataType, 'train',numFold)
                    angle_test  = Dataset4autoencoder(dataSetDir, dataType, 'test', numFold)
                    train_loader = DataLoader(angle_train, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(angle_test, batch_size=batch_size, shuffle=True)

                    # 시각화를 위한 tensorboard 초기화
                    writer_train = SummaryWriter(join(logDir,f'{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_embdim_{embedding_dim}/train/{numFold}_fold'))
                    writer_test =  SummaryWriter(join(logDir,f'{exp_name}/{modelVersion}/{nameDataset}/{dataType}/LR_{learningRate}_BS_{batch_size}_embdim_{embedding_dim}/test/{numFold}_fold'))
                    # 데이터 그래프 기록
                    my_model.eval()
                    x = torch.randn(1 ,seq_len, num_features, device=device)
                    writer_train.add_graph(my_model,x)
                    writer_test.add_graph(my_model,x)

                    for epoch in range(epochs):
                        # train session
                        my_model.train()
                        train_loss = 0
                        for batch_idx, (data) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                            # print(f'datainput shape:{data.shape}')
                            data = data.to(device)
                            optimizer.zero_grad()
                            output = my_model(data)
                            loss =  ((data - output)**2).sum() + my_model.encoder.kl 
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * data.size(0) # 이것은 모든 배치의 크기가 일정하지 않을 수 있기 때문에 이렇게 수행함! train_loss는 total loss of batch가 됨
                        train_loss /= len(train_loader.sampler)            
                        writer_train.add_scalar('loss(VAE)', train_loss, epoch)
                        # test session
                        test_loss = 0
                        my_model.eval()  # batch norm이나 dropout 등을 train mode 변환
                        with torch.no_grad():  # autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임
                            for data in test_loader:
                                data = data.to(device)
                                output = my_model(data)
                                loss =  ((data - output)**2).sum() + my_model.encoder.kl 
                                test_loss += loss.item() * data.size(0)
                            test_loss /= len(test_loader.sampler)
                            writer_test.add_scalar('loss(VAE)', test_loss, epoch)


                        print(f'\nTrain set: Average loss: {train_loss:.4f}, test set: Average loss: {test_loss:.4f}')
                    writer_train.add_hparams(
                            {"sess": "train", "Type": dataType, "lr": learningRate, "bsize": batch_size, "DS":nameDataset , 'lossFunc':lossFunction, 'emb_dim': embedding_dim}, 
                            { 
                                "loss": train_loss,
                            }, 
                        ) 
                    writer_test.add_hparams(
                            {"sess": "test",  "Type": dataType, "lr": learningRate, "bsize": batch_size, "DS":nameDataset, 'lossFunc':lossFunction, 'emb_dim': embedding_dim}, 
                            { 
                                "loss": test_loss,
                            }, 
                        ) 
                    writer_train.close()
                    writer_test.close()
                    dir_save_torch = join(SaveDir,modelVersion,nameDataset)
                    ensure_dir(dir_save_torch)
                    # 현재 아래 방법은 모델 저장이 안되는 단점이 있음
                    # model_scripted = torch.jit.script(my_model) # Export to TorchScript
                    # model_scripted.save(join(dir_save_torch,f'{dataType}_{numFold}_fold.pt')) # Save

                    # 수정된 방법
                    torch.save(my_model.state_dict(), join(dir_save_torch,f'{dataType}_{embedding_dim}_{numFold}_fold')) # 모델 정의가 필요함
                    # 수정된 방법으로 모델 불러올 떄
                    # Model class must be defined somewhere
                    # model.load_state_dict(torch.load(filepath))
                    # model.eval()

