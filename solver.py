import torch
import torch.optim as optim
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch_geometric.loader as data
from torch_geometric.datasets import Planetoid
from models import GraphConv, FullyConnected
from loss import LabelSmoothingLoss, Wasserstein_Metric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='citeseer', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.01,
                    help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=200,
                    help='Number of traing epoches')
parser.add_argument('--hidden_dim', type=list, default=2400,
                    help='Dimensions of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight for l2 loss on embedding matrix')
parser.add_argument('--log_interval', type=int,
                    default=10, help='Print iterval')
parser.add_argument('--log_dir', type=str, default='experiments',
                    help='Train/val loss and accuracy logs')
parser.add_argument('--checkpoint_interval', type=int,
                    default=20, help='Checkpoint saved interval')
parser.add_argument('--checkpoint_dir',
                    type=str, default='checkpoints',
                    help='Directory to save checkpoints'
                    )
args = parser.parse_args()
dataset = Planetoid(
    './data',
    'CiteSeer',
    transform=T.NormalizeFeatures()
)


gcn_model = GraphConv(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_dim,
    out_channels=dataset.num_classes,
    use_gdc=False,
    dropout=0.01,
    inference=0e-0,
)

fc_model = FullyConnected(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_dim,
    out_channels=dataset.num_classes
)

label_smoothing = LabelSmoothingLoss(num_classes=dataset.num_classes)
optimizer = optim.Adam([
    dict(params=gcn_model.conv1.parameters(), weight_decay=5e-4),
    dict(params=gcn_model.conv2.parameters(), weight_decay=0),
    dict(params=fc_model.parameters(), weight_decay=0e+0)
], lr=6e-4)

T = 10


def gcn_forward_pass(data, model):
    pred = model(data.x, data.edge_index, data.edge_weight)
    pred = pred[data.train_mask]
    truth = data.y[data.train_mask]
    return pred, F.cross_entropy(pred, truth)


def fc_forward_pass(data, model):
    pred = model(data.x)[data.train_mask]
    truth = data.y[data.train_mask]
    return pred, label_smoothing(pred, truth)


l_um = 0.3
l_ls = 0.001

for i in range(100):
    for batch_idx, batch in enumerate(data.DataLoader(dataset, batch_size=1)):
        optimizer.zero_grad()
        gcn_output, L_ce = gcn_forward_pass(batch, gcn_model)
        fc_output,  L_ls = fc_forward_pass(batch, fc_model)
        L_um = Wasserstein_Metric(gcn_output, fc_output)
        loss = L_ce + l_um * L_um + l_ls + L_ls
        loss.backward()
        optimizer.step()
        print('loss', loss.item(), 'ce', L_ce.item(),
              'ls', L_ls.item(), 'um', L_um.item())
