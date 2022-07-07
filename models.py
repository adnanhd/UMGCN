from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from variational_dropout import VariationalDropout
import torch


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = set(weights)
        self.dropout = dropout
        self.variational = variational

    def _drop_weights(self):
        state_dict = self.module.state_dict()
        for name_w in self.weights:
            w = None
            raw_w = state_dict[name_w]
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(
                    mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(
                    raw_w, p=self.dropout, training=self.training)
            state_dict[name_w] = w
        self.module.load_state_dict(state_dict)

    def forward(self, *args):
        self._drop_weights()
        return self.module.forward(*args)


class FullyConnected(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class GraphConv(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 use_gdc: bool,
                 dropout: float = 0.5,
                 inference: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             cached=True, normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             cached=True, normalize=not use_gdc)
        #self.conv1.lin = VariationalDropout(in_channels, hidden_channels)
        #self.conv2.lin = VariationalDropout(hidden_channels, out_channels)
        self.p = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = nn.functional.leaky_relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.softmax(x, dim=1)
