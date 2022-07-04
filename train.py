import torch
import torch.optim as optim
import torch_geometric.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from models import GraphConv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='citeseer', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.01,
                    help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=200,
                    help='Number of traing epoches')
parser.add_argument('--hidden_dim', type=list, default=16,
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


model = GraphConv(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_dim,
    out_channels=dataset.num_classes,
    use_gdc=False,
    p=args.dropout
)

optimizer = optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.init_lr)


def forward_pass(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()
