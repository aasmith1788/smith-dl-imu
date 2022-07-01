# 메인 아이디어
from CBDtorch.vaelstm_1layer import *  # VAE_LSTM layer바꿀 때 이것도 바꿀 것


class regressor(nn.Module):
    def __init__(self, filename, emb_dims, *args):
        super().__init__()
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = "cpu"
        self.VAE = RecurrentVariationalAutoencoder(*args)
        self.VAE.load_state_dict(torch.load(filename,map_location=map_location))

        self.dense = nn.Sequential(
            nn.Linear(emb_dims, 2048),
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

    def forward(self, x):
        x = self.VAE.encoder(x)
        output = self.dense(x)
        return output
