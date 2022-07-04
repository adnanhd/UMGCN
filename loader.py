import torch
from dataset import dataBuild


def load_dataset(cache=True):
    if cache:
        return torch.load('dataset.pt')
    else:
        return dataBuild('citeseer', './data/')
