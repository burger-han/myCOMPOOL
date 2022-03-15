import torch
import torch.nn
import torch.nn.functional as F
import dgl
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling
from pooling.topkpool import TopKPooling, get_batch_id, mytopk
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout


class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio:float, smallratio, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        if dgl.__version__ < "0.5":
            self.score_layer = GraphConv(in_dim, 1)
        else:
            self.score_layer = GraphConv(in_dim, 1, allow_zero_in_degree=True)
        self.non_linearity = non_linearity
        self.softmax = torch.nn.Softmax(dim=0)
        self.smallratio = float(smallratio)

    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, e_feat=None):
        score = self.score_layer(graph, feature).squeeze()
        perm, next_batch_num_nodes = mytopk(score, self.smallratio, self.ratio, get_batch_id(graph.batch_num_nodes()))

        next_batch_num_nodes_com= graph.batch_num_nodes()-next_batch_num_nodes
        # print("next_batch_num_nodes", next_batch_num_nodes)
        feature_dist = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph_dist = dgl.node_subgraph(graph, perm)

        mask = torch.ones(score.size(0)).to(score.device)
        mask[perm] = 0
        perm_com = torch.nonzero(mask).squeeze(-1).to(score.device)
        graph_com = dgl.node_subgraph(graph, perm_com)
        feature_com = feature[perm_com] * self.non_linearity(score[perm_com]).view(-1, 1)
        feature_full = feature * self.non_linearity(score).view(-1, 1)

        graph_dist.set_batch_num_nodes(next_batch_num_nodes)
        graph_com.set_batch_num_nodes(next_batch_num_nodes_com)

        score = self.softmax(score)
        if torch.nonzero(torch.isnan(score)).size(0) > 0:
            print(score[torch.nonzero(torch.isnan(score))])
            raise KeyError
        if e_feat is not None:
            e_feat = graph.edata['feat'].unsqueeze(-1)
        return graph_dist, graph_com, graph, feature_dist, feature_com, feature_full, perm, perm_com, score, e_feat


class SAGPoolReadout(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """
    def __init__(self, net_params, pool=True):
        super(SAGPoolReadout, self).__init__()
        in_dim = net_params['in_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_classes = net_params['n_classes']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.e_feat = net_params['edge_feat']

        self.conv = GCNLayer(in_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)
        self.conv1 = GCNLayer(in_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)
        self.conv2 = GCNLayer(out_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)
        self.conv3 = GCNLayer(out_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual)
        self.use_pool = pool
        self.pool = SAGPool(out_dim, ratio=net_params['pool_ratio'], smallratio=net_params['smallratio'])
        self.pool_global = SAGPool(out_dim*3, ratio=net_params['pool_ratio'], smallratio=net_params['smallratio'])
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.MLP_layer = MLPReadout(out_dim * 2, n_classes)
        self.MLP_layer_global = MLPReadout(out_dim * 6, n_classes)
        self.MLP_layer_avgpool = MLPReadout(out_dim, n_classes)

        self.MLP_layer_node = MLPReadout(out_dim, n_classes)
        self.MLP_layer_node_global = MLPReadout(out_dim*3, n_classes)
        self.loss_type = net_params['loss']

    def forward(self, graph, feature, e_feat=None):
        out1 = self.conv1(graph, feature)
        hg1 = torch.cat([self.avgpool(graph, out1), self.maxpool(graph, out1)], dim=-1)
        out2 = self.conv2(graph, out1)
        hg2 = torch.cat([self.avgpool(graph, out2), self.maxpool(graph, out2)], dim=-1)
        out3 = self.conv3(graph, out2)
        node_pred = self.MLP_layer_node(out3)  # 2 d
        if self.use_pool:
            graph_dist, graph_com, graph_full, out3_dist, out3_com, out3_full, perm, perm_com, node_score, e_feat = self.pool(graph, out3, e_feat)
        hg3 = torch.cat([self.avgpool(graph_dist, out3_dist), self.maxpool(graph_dist, out3_dist)], dim=-1)
        hg3_full = torch.cat([self.avgpool(graph_full, out3_full), self.maxpool(graph_full, out3_full)], dim=-1)
        hg_com = torch.cat([self.avgpool(graph_com, out3_com), self.maxpool(graph_com, out3_com)], dim=-1)
        hg = hg1 + hg2 + hg3
        hg_full = hg1 + hg2 + hg3_full

        scores = self.MLP_layer(hg)
        scores_com = self.MLP_layer(hg_com)
        scores_full = self.MLP_layer(hg_full)
        return scores, scores_com, scores_full, node_pred, node_score

    def forward_global(self, graph, feature, e_feat=None):
        out1 = self.conv1(graph, feature)
        out2 = self.conv2(graph, out1)
        out3 = self.conv3(graph, out2)
        out3 = torch.cat((out1,out2,out3), dim=-1)

        node_pred = self.MLP_layer_node_global(out3)

        if self.use_pool:
            graph_dist, graph_com, graph_full, out3_dist, out3_com, out3_full, perm, perm_com, node_score, e_feat = self.pool_global(graph, out3, e_feat)
        hg3 = torch.cat([self.avgpool(graph_dist, out3_dist), self.maxpool(graph_dist, out3_dist)], dim=-1)
        hg3_full = torch.cat([self.avgpool(graph_full, out3_full), self.maxpool(graph_full, out3_full)], dim=-1)
        hg3_com = torch.cat([self.avgpool(graph_com, out3_com), self.maxpool(graph_com, out3_com)], dim=-1)

        scores = self.MLP_layer_global(hg3)
        scores_com = self.MLP_layer_global(hg3_com)
        scores_full = self.MLP_layer_global(hg3_full)

        return scores, scores_com, scores_full, node_pred, node_score

    def forward_h(self, graph, feature, e_feat=None):
        out1 = self.conv1(graph, feature)
        if self.use_pool:
            graph_dist1, graph_com1, graph_full1, out1_dist, out1_com, out1_full, perm1, perm_com1, node_score1, e_feat = self.pool(
                graph, out1, e_feat)
        hg1 = torch.cat([self.avgpool(graph_dist1, out1_dist), self.maxpool(graph_dist1, out1_dist)], dim=-1)
        hg1_com = torch.cat([self.avgpool(graph_com1, out1_com), self.maxpool(graph_com1, out1_com)], dim=-1)
        hg1_full = torch.cat([self.avgpool(graph_full1, out1_full), self.maxpool(graph_full1, out1_full)], dim=-1)


        out2 = self.conv2(graph_dist1, out1_dist)
        if self.use_pool:
            graph_dist2, graph_com2, graph_full2, out2_dist, out2_com, out2_full, perm2, perm_com2, node_score2, e_feat = self.pool(
                graph_dist1, out2, e_feat)
        hg2 = torch.cat([self.avgpool(graph_dist2, out2_dist), self.maxpool(graph_dist2, out2_dist)], dim=-1)
        hg2_com = torch.cat([self.avgpool(graph_com2, out2_com), self.maxpool(graph_com2, out2_com)], dim=-1)
        hg2_full = torch.cat([self.avgpool(graph_full2, out2_full), self.maxpool(graph_full2, out2_full)], dim=-1)

        out3 = self.conv3(graph_dist2, out2_dist)
        if self.use_pool:
            graph_dist3, graph_com3, graph_full3, out3_dist, out3_com, out3_full, perm3, perm_com3, node_score3, e_feat = self.pool(graph_dist2, out3, e_feat)
        hg3 = torch.cat([self.avgpool(graph_dist3, out3_dist), self.maxpool(graph_dist3, out3_dist)], dim=-1)
        hg3_com = torch.cat([self.avgpool(graph_com3, out3_com), self.maxpool(graph_com3, out3_com)], dim=-1)
        hg3_full = torch.cat([self.avgpool(graph_full3, out3_full), self.maxpool(graph_full3, out3_full)], dim=-1)

        hg = hg1 + hg2 + hg3
        hg_com = hg1_com + hg2_com + hg3_com
        hg_full = hg1_full + hg2_full + hg3_full

        node_pred = self.MLP_layer_node(out1_full)
        scores = self.MLP_layer(hg)
        scores_com = self.MLP_layer(hg_com)
        scores_full = self.MLP_layer(hg_full)
        return scores, scores_com, scores_full, node_pred, node_score1

    def forward2(self, graph, feature, e_feat=None):
        #  readout function变成average mean pool，输出的node pred和graph pred在同一个domain。
        out = self.conv(graph, feature)
        if self.use_pool:
            graph_dist, graph_com, feature_dist, feature_com, feature_full,  perm, perm_com, node_score, e_feat = self.pool(graph, out, e_feat)
        hg = self.avgpool(graph_dist, feature_dist)
        hg_com = self.avgpool(graph_com, feature_com)

        scores = self.MLP_layer_avgpool(hg)
        scores_com = self.MLP_layer_avgpool(hg_com)
        node_pred = self.MLP_layer_avgpool(feature_full)
        return scores, scores_com,  node_pred, node_score

    def loss(self, pred, label, cluster=False):
        if self.loss_type == 'cross':
            criterion = torch.nn.CrossEntropyLoss()
            loss_cls = criterion(pred, label.long())
        else:
            loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
        return loss_cls
