import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from utils import *
from logger import Logger
import os.path as osp
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm


    
class HLGNN(MessagePassing):
    def __init__(self, data, args):
        super(HLGNN, self).__init__(aggr='add')
        self.K = args.K
        self.init = args.init
        self.alpha = args.alpha
        self.dropout = args.dropout
        self.lin1 = Linear(data.num_features, data.num_features)

        assert self.init in ['SGC', 'RWR', 'KI', 'Random']
        if self.init == 'SGC':
            alpha = int(self.alpha)
            TEMP = 0.0 * np.ones(self.K+1)
            TEMP[alpha] = 1.0
        elif self.init == 'RWR':
            TEMP = self.alpha * (1-self.alpha) ** np.arange(self.K+1)
            TEMP[-1] = (1-self.alpha) ** self.K
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif self.init == 'KI':
            TEMP = self.alpha ** np.arange(self.K+1)
            # TEMP = TEMP / np.sum(np.abs(TEMP))
        elif self.init == 'Random':
            bound = np.sqrt(3 / (self.K+1))
            TEMP = np.random.uniform(-bound, bound, self.K+1)
            TEMP = TEMP / np.sum(np.abs(TEMP))

        self.temp = Parameter(torch.tensor(TEMP))
        # self.beta = Parameter(torch.zeros(3))
        
    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.init == 'SGC':
            self.alpha = int(self.alpha)
            self.temp.data[self.alpha]= 1.0
        elif self.init == 'RWR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha * (1-self.alpha) ** k
            self.temp.data[-1] = (1-self.alpha) ** self.K
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.init == 'KI':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha ** k
            # self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.init == 'Random':
            bound = np.sqrt(3 / (self.K+1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))


    def forward(self, x, adj_t, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        adj_t = gcn_norm(adj_t, edge_weight, adj_t.size(0), dtype=torch.float)
        # edge_index, row_n = row_norm(raw_edge_index, edge_weight, num_nodes, dtype=torch.float)
        # edge_index, column_n = column_norm(raw_edge_index, edge_weight, num_nodes, dtype=torch.float)
        
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(adj_t, x=x, edge_weight=edge_weight, size=None)
            gamma = self.temp[k+1]
            hidden = hidden + gamma * x
        return hidden
    
    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j
    
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce="add")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
    