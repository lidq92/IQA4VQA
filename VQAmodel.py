import torch
import torch.nn as nn
from torchvision import models


class QAModel(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True, pool_mode='mean', reduced_size=256, hidden_size=64, num_layers=2):
        super(QAModel, self).__init__()
        features = list(models.__dict__[arch](pretrained=pretrained).children())[:-2]
        self.features = nn.Sequential(*features)
        assert('res' in arch) # ResNet-like arch
        input_size = features[-1][-1].conv1.in_channels # 
        if pool_mode == 'mean+std':
            input_size = 2 * input_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dr = nn.Sequential(nn.Linear(input_size, reduced_size), nn.LayerNorm(reduced_size)) 
        self.fp = nn.GRU(reduced_size, hidden_size, num_layers, batch_first=True)
        self.regression = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, pool_features):
        # pool_features: N x T x C
        N = pool_features.size(0)
        df = self.dr(pool_features) # N x T x reduced_size
        pf, _ = self.fp(df, self._get_initial_state(N, df.device)) # N x T x hidden_size
        q = self.regression(pf) # N x T x 1
        Q = q.mean(dim=1) # N x 1
        return Q
    

class WholeQAModel(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True, pool_mode='mean', reduced_size=256, hidden_size=64, num_layers=2):
        super(WholeQAModel, self).__init__()
        features = list(models.__dict__[arch](pretrained=pretrained).children())[:-2]
        self.features = nn.Sequential(*features)
        assert('res' in arch) # ResNet-like arch
        input_size = features[-1][-1].conv1.in_channels # 
        if pool_mode == 'mean+std':
            input_size = 2 * input_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dr = nn.Sequential(nn.Linear(input_size, reduced_size), nn.LayerNorm(reduced_size)) 
        self.fp = nn.GRU(reduced_size, hidden_size, num_layers, batch_first=True)
        self.regression = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pool_mode = pool_mode
        
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, x): # x: N x T x 3 x H x W
        N, T, _, H, W = x.size()
        x = x.view(-1, 3, H, W) # NT x 3 x H x W
        f = self.features(x) # NT x input_size x H1 x W1
        m1 = self.pool(f).view(N, T, -1) # N x T x input_size
        if self.pool_mode == 'mean':
            pool_features = m1
        else:
            rm2 = torch.sqrt(F.relu(self.pool(torch.pow(f, 2)).view(N, T, -1) - torch.pow(m1, 2)))
            if self.pool_mode == 'std':
                pool_features = rm2
            elif self.pool_mode == 'mean+std':
                pool_features = torch.cat((m1, rm2), 2) # statistical pooling: mean & std
            else:
                print('Unknown pool mode!')
        df = self.dr(pool_features) # N x T x reduced_size
        pf, _ = self.fp(df, self._get_initial_state(N, df.device)) # N x T x hidden_size
        q = self.regression(pf) # N x T x 1
        Q = q.mean(dim=1) # N x 1
        
        return Q
