# 메인 아이디어
from CBDtorch.vaelstm_1layer import *  # VAE_LSTM layer바꿀 때 이것도 바꿀 것


class regressor(nn.Module):
    def __init__(self, filename, emb_dims, *args):
        self.emb_dims = emb_dims
        super().__init__()
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = "cpu"
        self.VAE = RecurrentVariationalAutoencoder(*args)
        self.VAE.load_state_dict(torch.load(filename, map_location=map_location))

        self.dense = nn.Sequential(
            nn.Linear(emb_dims * 4, 2048),  # 압축된 feature + demograpchic 정보 3개를 가로로 추가
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 노드를 학습과정에서 얼만큼 활용 안할지
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 303),
        )
        ##You can use as many linear layers and other activations as you want

    def forward(self, x_imu, x_dg):
        compressed_features = self.VAE.encoder(x_imu)
        dg_features = x_dg.repeat(1, self.emb_dims)
        concated = torch.cat(
            (compressed_features, dg_features), 1
        )  # [batchsize, 압축피쳐] + [batchzie, dg피쳐 3개(압축피쳐 3개분)] = [batchsize, 합_피쳐(총4개)]
        output = self.dense(concated)
        return output
