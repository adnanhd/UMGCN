from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch


# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, acti=True):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim)  # bias = False is also ok.
#         if acti:
#             self.acti = nn.ReLU(inplace=True)
#         else:
#             self.acti = None
#
#     def forward(self, F):
#         output = self.linear(F)
#         if not self.acti:
#             return output
#         return self.acti(output)


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
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
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNConv(input_dim, hidden_dim)
        self.gcn_layer2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        X = self.dropout(X)
        print(A.dtype, X.dtype)
        F = torch.mm(A, X)
        F = self.gcn_layer1(F)
        F = nn.functional.relu(F)
        F = self.dropout(F)
        F = torch.mm(A, F)
        output = self.gcn_layer2(F)
        return output


class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_gdc, p=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             cached=True, normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             cached=True, normalize=not use_gdc)
        self.p = p

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = nn.functional.leaky_relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
