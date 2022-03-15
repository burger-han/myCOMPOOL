import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling
from pooling.topkpool import TopKPooling, get_batch_id
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
from nets.compool import *


class COMSAGPool_b(torch.nn.Module):
    def __init__(self, in_dim:int, ratio, smallratio, non_linearity=torch.tanh):
        super(COMSAGPool_b, self).__init__()
        self.in_dim = in_dim
        self.ratio = float(ratio)
        self.com_ratio = 1 - float(ratio)
        self.smallratio = float(smallratio)
        if dgl.__version__ < "0.5":
            self.score_layer = GraphConv(in_dim, 1)
        else:
            self.score_layer = GraphConv(in_dim, 1, allow_zero_in_degree=True)
        self.non_linearity = non_linearity
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, e_feat=None):
        score = self.score_layer(graph, feature).squeeze()
        perm, k = mytopk(score, self.smallratio, self.ratio, get_batch_id(graph.batch_num_nodes()))

    #   得到perm——com
        mask = torch.ones(score.size(0)).to(score.device)
        mask[perm] = 0
        perm_com = torch.nonzero(mask).squeeze(-1).to(score.device)

        feature_dis = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        feature_com = feature[perm_com] * self.non_linearity(score[perm_com]).view(-1, 1)

        graph_dis = dgl.node_subgraph(graph, perm)
        graph_com = dgl.node_subgraph(graph, perm_com)

        graph_dis.set_batch_num_nodes(k)
        k_com = graph.batch_num_nodes()-k
        graph_com.set_batch_num_nodes(k_com)

        score = self.softmax(score)
        if torch.nonzero(torch.isnan(score)).size(0) > 0:
            print('score', score)
            print(score[torch.nonzero(torch.isnan(score))])
            raise KeyError
        if e_feat is not None:
            e_feat = graph_dis.edata['feat'].unsqueeze(-1)
            e_feat_com = graph_com.edata['feat'].unsqueeze(-1)
        return graph_dis,graph_com,feature_dis,feature_com,perm,perm_com,score,e_feat

# class SAGPool_b(torch.nn.Module):
#     def __init__(self, in_dim:int, ratio, smallratio, non_linearity=torch.tanh):
#         super(SAGPool_b, self).__init__()
#         self.in_dim = in_dim
#         self.smallratio = float(smallratio)
#         self.ratio = ratio
#         if dgl.__version__ < "0.5":
#             self.score_layer = GraphConv(in_dim, 1)
#         else:
#             self.score_layer = GraphConv(in_dim, 1, allow_zero_in_degree=True)
#         self.non_linearity = non_linearity
#         self.softmax = torch.nn.Softmax()
#
#     def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, e_feat=None):
#         score = self.score_layer(graph, feature).squeeze()
#         #print("feature", feature.shape)
#         perm, next_batch_num_nodes = TopKPooling(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
#         feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
#         graph = dgl.node_subgraph(graph, perm)
#
#         graph.set_batch_num_nodes(next_batch_num_nodes)
#
#         score = self.softmax(score)
#         if torch.nonzero(torch.isnan(score)).size(0) > 0:
#             print(score[torch.nonzero(torch.isnan(score))])
#             raise KeyError
#
#         if e_feat is not None:
#             e_feat = graph.edata['feat'].unsqueeze(-1)
#
#         return graph, feature, perm, score, e_feat


class SAGPoolReadout_b(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """
    def __init__(self, net_params, pool=True):
        super(SAGPoolReadout_b, self).__init__()
        in_dim = net_params['in_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_classes = net_params['n_classes']
        pool_ratio = net_params['pool_ratio']
        smallratio = net_params['smallratio']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.e_feat = net_params['edge_feat']
        self.conv1 = GCNLayer(in_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)
        self.conv2 = GCNLayer(out_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)
        self.conv3 = GCNLayer(out_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)

        self.use_pool = pool
        #self.pool = SAGPool_b(out_dim * 3, ratio=pool_ratio, smallratio=smallratio)
        self.pool = COMSAGPool_b(out_dim * 3, ratio=pool_ratio, smallratio=smallratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.MLP_layer = MLPReadout(out_dim * 6, n_classes)
        self.mse = torch.nn.MSELoss()

    def forward(self, graph, feature, e_feat=None):
        out1 = self.conv1(graph, feature)
        out2 = self.conv2(graph, out1)
        out3 = self.conv3(graph, out2)
        out = torch.cat((out1, out2, out3), dim=-1)
        if self.use_pool:
            graph, graph_com, out, out_com, _, _, _, _ = self.pool(graph, out, e_feat)
        hg3 = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        hg3_com = torch.cat([self.avgpool(graph_com, out_com), self.maxpool(graph_com, out_com)], dim=-1)

        scores = self.MLP_layer(hg3)
        hg3_com = self.MLP_layer(hg3_com) #降维做loss

        center_com = torch.mean(hg3_com, dim=0).unsqueeze(0).repeat(hg3_com .shape[0], 1)
        mse_loss = self.mse(center_com, hg3_com)

        #scores_com = self.MLP_layer(hg3_com)
        return scores, mse_loss

    def loss(self, pred, label, cluster=False):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, label.long())
        return loss