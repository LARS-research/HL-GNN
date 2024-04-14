# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, TransformerConv, GATConv
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter, Linear
from utils import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul

class BaseGNN(torch.nn.Module):
    def __init__(self, dropout, num_layers):
        super(BaseGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_weight):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.num_layers == 1:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class HLGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout, alpha, init):
        super(HLGNN, self).__init__(aggr='add')
        self.K = K
        self.init = init
        self.alpha = alpha
        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)

        assert init in ['SGC', 'RWR', 'KI', 'Random']
        if init == 'SGC':
            alpha = int(alpha)
            TEMP = 0.0 * np.ones(K+1)
            TEMP[alpha] = 1.0
        elif init == 'RWR':
            TEMP = alpha * (1-alpha) ** np.arange(K+1)
            TEMP[-1] = (1-alpha) ** K
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif init == 'KI':
            TEMP = alpha ** np.arange(K+1)
            # TEMP = TEMP / np.sum(np.abs(TEMP))
        elif init == 'Random':
            bound = np.sqrt(3 / (K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
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

class SAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(first_channels, second_channels))


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(first_channels, second_channels, normalize=False))
            
class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GAT, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GATConv(first_channels, second_channels))

class WSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(WSAGE, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GraphConv(first_channels, second_channels))


class Transformer(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Transformer, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(TransformerConv(first_channels, second_channels))


class MLPPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
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
        return x


class MLPCatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPCatPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        in_channels = 2 * in_channels
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x1 = torch.cat([x_i, x_j], dim=-1)
        x2 = torch.cat([x_j, x_i], dim=-1)
        for lin in self.lins[:-1]:
            x1, x2 = lin(x1), lin(x2)
            x1, x2 = F.relu(x1), F.relu(x2)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.lins[-1](x1)
        x2 = self.lins[-1](x2)
        x = (x1 + x2)/2
        return x


class MLPDotPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLPDotPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = F.relu(x_i), F.relu(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(x_i * x_j, dim=-1)
        return x



class DotPredictor(torch.nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class BilinearPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BilinearPredictor, self).__init__()
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)

    def reset_parameters(self):
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x


