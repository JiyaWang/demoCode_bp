'''This file is to load and manage datasets(Cora, CiteSeer, PubMed and Photo)'''

import os.path as osp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Planetoid, Twitch

def get_planetoid_dataset(name, normalize_features=True, transform=None, split="complete"):
    path = osp.join('.', 'data', name)
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, name, split=split)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y,dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        
        if split=='train':
            self.mask = dataset[0].train_mask.to(device)
        if split=='val':
            self.mask = dataset[0].val_mask.to(device)
        if split=='test':
            self.mask = dataset[0].test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.edge_index
    

def get_amazon_dataset(name, normalize_features=True, transform=None, split="complete"):#split="complete"
    path = osp.join('.', 'data', name)
    dataset = Amazon(path, name)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

def get_Twitch_dataset(name, normalize_features=True, transform=None, split="complete"):#split="complete"
    path = osp.join('.', 'data', name)
    dataset = Twitch(path, name)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Photo', device='cuda'):
        dataset = get_amazon_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y, dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[:dataset[0].num_nodes - 1000] = 1
        val_mask.fill_(False)
        val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        test_mask.fill_(False)
        test_mask[dataset[0].num_nodes - 500:] = 1
        if split == 'train':
            self.mask = train_mask.to(device)
        if split == 'val':
            self.mask = val_mask.to(device)
        if split == 'test':
            self.mask = test_mask.to(device)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
            return self.samples_per_epoch

    def __getitem__(self, idx):
            return self.X, self.y, self.mask, self.edge_index
    


class TwitchDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='DE', device='cpu'):
        dataset = get_Twitch_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y, dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[:dataset[0].num_nodes - 1000] = 1
        val_mask.fill_(False)
        val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        test_mask.fill_(False)
        test_mask[dataset[0].num_nodes - 500:] = 1
        if split == 'train':
            self.mask = train_mask.to(device)
        if split == 'val':
            self.mask = val_mask.to(device)
        if split == 'test':
            self.mask = test_mask.to(device)

        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
            return self.samples_per_epoch

    def __getitem__(self, idx):
            return self.X, self.y, self.mask, self.edge_index