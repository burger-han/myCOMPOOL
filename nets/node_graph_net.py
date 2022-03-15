import torch
from torch_geometric.nn import GCNConv, WLConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import scatter_add, scatter_max
import torch.nn.functional as F
import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling
from pooling.topkpool import TopKPooling, get_batch_id
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class NodeGraphNet(torch.nn.Module):
    def __init__(self, net_params):
        super(NodeGraphNet, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        n_layers = net_params['L']
        self.in_feat_dropout = nn.Dropout(net_params['in_feat_dropout'])
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.mse_loss = torch.nn.MSELoss()
        self.non_linearity = torch.tanh
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.MLP_layer = MLPReadout(2 * hidden_dim, n_classes)
        self.MLP_layer_node = MLPReadout(3 * hidden_dim, n_classes)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph:dgl.DGLGraph, labels, feature:torch.Tensor, e_feat=None):
        x = self.embedding_h(feature)
        x = self.in_feat_dropout(x)
        for conv in self.layers:
            x = conv(graph, x)
        graph.ndata['h'] = x
        hg = torch.cat([self.avgpool(graph, x), self.maxpool(graph, x)], dim=-1)
        x_g = torch.cat((torch.repeat_interleave(hg, graph.batch_num_nodes(), dim=0), x), dim=1)
        node_label = torch.repeat_interleave(labels, graph.batch_num_nodes(), dim=0)

        #   GCN
        g = self.MLP_layer(hg)
        x_g = self.MLP_layer_node(x_g)

        #   plot_margin(g_before_softmax, data.y)
        return x, x_g, g, node_label

    def loss(self, graph_pred, node_pred, label, node_label):
        #criterion = nn.CrossEntropyLoss()
        #loss = criterion(pred, label)
       # print(node_label.shape, node_pred.shape)
        cls_loss_node = F.nll_loss(node_pred, node_label.long())
        cls_loss_graph = F.nll_loss(graph_pred, label.long())
        loss = cls_loss_node + cls_loss_graph
        return loss