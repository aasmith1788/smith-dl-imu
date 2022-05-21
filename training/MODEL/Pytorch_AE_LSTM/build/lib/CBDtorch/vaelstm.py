# 필요한 라이브러리 설정
import torch
from torch import nn


class VariationalEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, device):
        super(VariationalEncoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # bidirectianl이 켜져 있어서 그럼
        self.mu = torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.sigma = torch.nn.Linear(
            self.embedding_dim * 2, self.embedding_dim)  # bidirectianl이 켜져 있어서 그럼
        self.N = torch.distributions.Normal(0, 1)
        # cuda()
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        if device == 'cpu':
            self.N.loc = self.N.loc.cpu()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cpu()
        self.kl = 0

    def forward(self, x):
        # print(f'input:{x.shape}')
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'afterlstm:{x.shape}')
        mu = self.mu(x[:, -1, :])
        sigma = torch.exp(self.sigma(x[:, -1, :]))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

# https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/25
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x_reshape = x.contiguous().view(t * n, -1)  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, -1)  # (samples, timesteps, output_size)
        return y

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True,
          bidirectional = True
        )
        self.rnn2 = nn.LSTM(
          input_size=input_dim * 2,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True,
          bidirectional = True
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim * 2, self.n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, x):
        # print(f'decoder first shape of x: {x.shape}')
        x = x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)
        # print(f'decoder after repeatvector shape of x: {x.shape}')       
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # print(f'decoder last shape of x: {self.timedist(x).shape}')
        return self.timedist(x)

# main module
class RecurrentVariationalAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=30, device='cuda'):
        super(RecurrentVariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(seq_len, n_features, embedding_dim, device).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
    def forward(self, x):
        # print(f'first shape of x: {x.shape}')
        z = self.encoder(x)
        # print(f'last shape of x: {x.shape}')
        return self.decoder(z)