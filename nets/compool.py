import torch
import torch.nn
import torch.nn.functional as F
import math
import dgl
from ultils import *
from dgl.nn import GraphConv, AvgPooling, MaxPooling
from pooling.topkpool import TopKPooling, get_batch_id, mytopk
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
import numpy as np

def plot_scatter_singlegraph(g, y, label, vis_dir):
    y1_number = (y==1).nonzero().size(0)
    y0_number = (y==0).nonzero().size(0)
    g = g.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    fig, ax = plt.subplots()
    # Scatter plot of data colored with labels
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.scatter(g[:, 0], g[:, 1], c=y, s=6, cmap=plt.cm.Paired)
    number_of_nodes = g.shape[0]
    name = vis_dir.split('/')[-1]
    title = name+' node number{}'.format(number_of_nodes)+'useful_node:{} label :{}'.format(y1_number, label)
    plt.title(title)
    plt.savefig(vis_dir)
    plt.close('all')
    return

def get_bias_score(graph:dgl.DGLGraph, feature, label):
    node_label = torch.repeat_interleave(label, graph.batch_num_nodes(), dim=0)
    avg = AvgPooling()
    graph_embedding = avg(graph, feature)
    idx_cls0 = (label == 0).nonzero().squeeze(-1).long()
    idx_cls1 = (label == 1).nonzero().squeeze(-1).long()

    center0 = torch.mean(graph_embedding[idx_cls0], dim=0)
    center1 = torch.mean(graph_embedding[idx_cls1], dim=0)

    distance_center0 = torch.sqrt(torch.sum((feature[:,None,:]-center0)**2, dim=2)).squeeze(-1)
    distance_center1 = torch.sqrt(torch.sum((feature[:,None,:]-center1)**2, dim=2)).squeeze(-1)

    #print(distance_center0, distance_center1)
    if idx_cls0.size(0) == 0:
        score_distance = distance_center1
    elif idx_cls1.size(0) == 0:
        score_distance = distance_center0

    elif idx_cls0.size(0) != 0 and idx_cls1.size(0) != 0:
        #print("distance_center0 - distance_center1", (distance_center0-distance_center1).shape)
        score_distance = torch.mul((distance_center0 - distance_center1),(node_label - 0.5) * 2)
        # print(node_label.shape, (distance_center0 - distance_center1).shape)
        # print(score_distance)

    return score_distance


class COMSAGPool(torch.nn.Module):
    def __init__(self, in_dim:int, ratio, smallratio, non_linearity=torch.tanh):
        super(COMSAGPool, self).__init__()
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

    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, label, e_feat=None):
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

class COMSAGPool_multi_scores(torch.nn.Module):
    def __init__(self, in_dim:int, ratio, smallratio, non_linearity=torch.tanh):
        super(COMSAGPool_multi_scores, self).__init__()
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
        self.score_mlp = MLPReadout(2, 1)
        self.lin1 = torch.nn.Linear(2, 1)

    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, label, e_feat=None):

        score_distance = get_bias_score(graph, feature, label)
        score_gcn = self.score_layer(graph, feature).squeeze()
        #print("shape of 2 score", score_distance.shape, score_gcn.shape,score_distance.unsqueeze(-1).shape, score_gcn.shape)

        score = torch.cat((score_gcn.unsqueeze(-1),score_distance.unsqueeze(-1)),dim=-1)
        print("score shape before mlp", score.shape)

        score = self.lin1(score)
        print("score_gcn.shape,score.shape ",score_gcn.shape,score.shape )
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

class COMSAGPool_gumble(torch.nn.Module):
    def __init__(self, in_dim:int, ratio, smallratio, non_linearity=torch.tanh):
        super(COMSAGPool_gumble, self).__init__()
        self.in_dim = in_dim
        self.ratio = float(ratio)
        self.com_ratio = 1 - float(ratio)
        self.smallratio = float(smallratio)
        if dgl.__version__ < "0.5":
            self.conv1 = GraphConv(in_dim, in_dim)
            self.conv2 = GraphConv(in_dim, 2)
        else:
            self.conv1 = GraphConv(in_dim, in_dim, allow_zero_in_degree=True)
            self.conv2 = GraphConv(in_dim, 2, allow_zero_in_degree=True)
        self.non_linearity = non_linearity
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, label, e_feat=None):
        out1 = self.conv1(graph, feature)
        out2 = self.conv2(graph, out1)
        sample = F.gumbel_softmax(out2, hard=True)
        perm1 = sample[:, 0].nonzero()
        perm2 = sample[:, 1].nonzero()

        perm = perm1.squeeze(-1)
        p_useful = torch.softmax(out2, dim=-1)[:, 0][perm1]


        perm_com = perm2.squeeze(-1)
        p_useless = torch.softmax(out2, dim=-1)[:, 1][perm2]
        #print("perm", perm1.shape, perm2.shape, sample.shape)

        feature_dis = feature[perm] * self.non_linearity(p_useful).view(-1, 1)
        feature_com = feature[perm_com] * self.non_linearity(p_useless).view(-1, 1)

        graph_dis = dgl.node_subgraph(graph, perm)
        graph_com = dgl.node_subgraph(graph, perm_com)
        # graph_dis.set_batch_num_nodes(k)
        # k_com = graph.batch_num_nodes()-k
        # graph_com.set_batch_num_nodes(k_com)

        if e_feat is not None:
            e_feat = graph_dis.edata['feat'].unsqueeze(-1)
            e_feat_com = graph_com.edata['feat'].unsqueeze(-1)
        return perm,perm_com,feature_dis,feature_com,perm,perm_com,None,e_feat


class COMPoolNet(torch.nn.Module):
    def __init__(self, net_params, pool=True):
        super(COMPoolNet, self).__init__()
        in_dim = net_params['in_dim']
        out_dim =net_params['out_dim']
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
        #self.pool = COMSAGPool(out_dim, ratio=pool_ratio, smallratio=smallratio)
        self.pool = COMSAGPool_multi_scores(out_dim, ratio=pool_ratio, smallratio=smallratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.MLP_layer = MLPReadout(out_dim * 2, n_classes)
        self.MLP_layer_node = MLPReadout(out_dim, n_classes)
        self.mse = torch.nn.MSELoss()
        self.loss_type = net_params['loss']

    def forward(self, graph, feature, label, e_feat=None):
        out1 = self.conv1(graph, feature)
        hg1 = torch.cat([self.avgpool(graph, out1), self.maxpool(graph, out1)], dim=-1)
        out2 = self.conv2(graph, out1)
        hg2 = torch.cat([self.avgpool(graph, out2), self.maxpool(graph, out2)], dim=-1)
        out3 = self.conv3(graph, out2)

        # sample = F.gumbel_softmax(out3, hard=True)
        # print("sample", sample.shape)
        # print("sample", torch.argmax(sample, dim=-1))

        # node_pred = self.conv_node(graph, out3)
        node_pred = self.MLP_layer_node(out3)
        if self.use_pool:
            graph, graph_com, out3, out_com, _, _, scores, e_feat = self.pool(graph, out3, label, e_feat)
        hg3 = torch.cat([self.avgpool(graph, out3), self.maxpool(graph, out3)], dim=-1)
        hg3_com = torch.cat([self.avgpool(graph_com, out_com), self.maxpool(graph_com, out_com)], dim=-1)


        hg = hg1 + hg2 + hg3
        pred = self.MLP_layer(hg)
        pred_com = self.MLP_layer(hg3_com)
        corr_idx = (torch.argmax(pred, dim=-1) == label).nonzero().squeeze(-1).long()
        return pred, pred_com, hg3_com, node_pred, scores # node_pred: 2d, scores: 1d 且是两层gcn之后的score

    def loss(self, pred, label, hg_com):
        center = torch.mean(hg_com, dim=0).unsqueeze(0).repeat(hg_com.shape[0], 1)
        loss_com = self.mse(center, hg_com).item()
        if self.loss_type == 'cross':
            criterion = torch.nn.CrossEntropyLoss()
            loss_cls = criterion(pred, label.long())
        else:
            loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
        return loss_cls, loss_com

class COMPoolNet_gumble(torch.nn.Module):
    def __init__(self, net_params, pool=True):
        super(COMPoolNet_gumble, self).__init__()
        in_dim = net_params['in_dim']
        out_dim =net_params['out_dim']
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
        self.score_layer = GCNLayer(out_dim, n_classes, F.relu, dropout, self.batch_norm, self.residual)
        self.use_pool = pool
        self.pool = COMSAGPool_gumble(out_dim, ratio=pool_ratio, smallratio=smallratio)
        self.pool_init = COMSAGPool(out_dim, ratio=pool_ratio, smallratio=smallratio)

        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.MLP_layer = MLPReadout(out_dim * 2, n_classes)
        self.MLP_layer_node = MLPReadout(out_dim, n_classes)
        self.mse = torch.nn.MSELoss()
        self.loss_type = net_params['loss']

    def forward(self, graph, feature, label, e_feat=None):
        out1 = self.conv1(graph, feature)
        hg1 = torch.cat([self.avgpool(graph, out1), self.maxpool(graph, out1)], dim=-1)
        out2 = self.conv2(graph, out1)
        hg2 = torch.cat([self.avgpool(graph, out2), self.maxpool(graph, out2)], dim=-1)
        out3 = self.conv3(graph, out2)

        #out3 = out1 + out2 + out3

        node_pred = self.MLP_layer_node(out3)
        if self.use_pool:
            perm, perm_com, out_dist, out_com, _, _, scores, e_feat = self.pool(graph, out3, label, e_feat)
        #out3[perm] = out_dist
        out3[perm_com] = out3[perm_com] * 0.001
        hg3= torch.cat([self.avgpool(graph, out3), self.maxpool(graph, out3)], dim=-1)

        hg3_com = hg3

        hg = hg1 + hg2 + hg3
        pred = self.MLP_layer(hg)
        pred_com = self.MLP_layer(hg3_com)

        return pred, pred_com, hg3_com, node_pred, scores # node_pred: 2d, scores: 1d 且是两层gcn之后的score

    def loss(self, pred, label, hg_com):
        #if hg_com != None:
        center = torch.mean(hg_com, dim=0).unsqueeze(0).repeat(hg_com.shape[0], 1)
        loss_com = self.mse(center, hg_com).item()
        #else:

        if self.loss_type == 'cross':
            criterion = torch.nn.CrossEntropyLoss()
            loss_cls = criterion(pred, label.long())
        else:
            loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
        return loss_cls, loss_com

class COMPoolNet_h(torch.nn.Module):
    def __init__(self, net_params, pool=True):
        super(COMPoolNet_h, self).__init__()
        in_dim = net_params['in_dim']
        out_dim =net_params['out_dim']
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
        self.conv_node = GCNLayer(out_dim, n_classes, F.relu, dropout, self.batch_norm, self.residual)
        self.use_pool = pool
        self.pool = COMSAGPool(out_dim, ratio=pool_ratio, smallratio=smallratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.MLP_layer = MLPReadout(out_dim * 2, n_classes)

        self.MLP_layer_node = MLPReadout(out_dim, n_classes)
        self.mse = torch.nn.MSELoss()
        self.loss_type = net_params['loss']

    def forward(self, graph, feature, label,  e_feat=None):
        out1 = self.conv1(graph, feature)
        if self.use_pool:
            graph_dist1, graph_com1, out1_dist, out1_com, perm1, perm_com1, node_score1, e_feat = self.pool(
                graph, out1, e_feat)
        hg1 = torch.cat([self.avgpool(graph_dist1, out1_dist), self.maxpool(graph_dist1, out1_dist)], dim=-1)
        hg1_com = torch.cat([self.avgpool(graph_com1, out1_com), self.maxpool(graph_com1, out1_com)], dim=-1)
        out2 = self.conv2(graph_dist1, out1_dist)
        if self.use_pool:
            graph_dist2, graph_com2, out2_dist, out2_com, perm2, perm_com2, node_score2, e_feat = self.pool(
                graph_dist1, out2, e_feat)
        hg2 = torch.cat([self.avgpool(graph_dist2, out2_dist), self.maxpool(graph_dist2, out2_dist)], dim=-1)
        hg2_com = torch.cat([self.avgpool(graph_com2, out2_com), self.maxpool(graph_com2, out2_com)], dim=-1)
        out3 = self.conv3(graph_dist2, out2_dist)
        if self.use_pool:
            graph_dist3, graph_com3, out3_dist, out3_com, perm3, perm_com3, node_score3, e_feat = self.pool(graph_dist2, out3, e_feat)
        hg3 = torch.cat([self.avgpool(graph_dist3, out3_dist), self.maxpool(graph_dist3, out3_dist)], dim=-1)
        hg3_com = torch.cat([self.avgpool(graph_com3, out3_com), self.maxpool(graph_com3, out3_com)], dim=-1)

        hg = hg1 + hg2 + hg3
        hg_com = hg1_com + hg2_com + hg3_com

        node_pred = self.MLP_layer_node(out3)
        scores = self.MLP_layer(hg)
        scores_com = self.MLP_layer(hg_com)
        return scores, scores_com, hg3_com, node_pred, node_score1

    def loss(self, pred, label, hg_com):
        center = torch.mean(hg_com, dim=0).unsqueeze(0).repeat(hg_com.shape[0], 1)
        loss_com = self.mse(center, hg_com).item()
        if self.loss_type == 'cross':
            criterion = torch.nn.CrossEntropyLoss()
            loss_cls = criterion(pred, label.long())
        else:
            loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
        return loss_cls, loss_com


class COMPoolNet_global(torch.nn.Module):
    def __init__(self, net_params, pool=True):
        super(COMPoolNet_global, self).__init__()
        in_dim = net_params['in_dim']
        out_dim =net_params['out_dim']
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
        self.pool = COMSAGPool(out_dim*3, ratio=pool_ratio, smallratio=smallratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        self.MLP_layer_global = MLPReadout(out_dim * 6, n_classes)
        self.MLP_layer_node_global = MLPReadout(out_dim*3, n_classes)
        self.MLP_layer_com = MLPReadout(out_dim * 2, n_classes)
        self.mse = torch.nn.MSELoss()
        self.loss_type = net_params['loss']

    def forward(self, graph, feature,label,  e_feat=None):
        out1 = self.conv1(graph, feature)
        out2 = self.conv2(graph, out1)
        out3 = self.conv3(graph, out2)
        out3 = torch.cat((out1, out2, out3), dim=-1)
        node_pred = self.MLP_layer_node_global(out3)
        if self.use_pool:
            graph_dist, graph_com, out3_dist, out3_com, perm, perm_com, node_score, e_feat = self.pool(graph, out3, label, e_feat)
        hg3 = torch.cat([self.avgpool(graph_dist, out3_dist), self.maxpool(graph_dist, out3_dist)], dim=-1)
        hg3_com = torch.cat([self.avgpool(graph_com, out3_com), self.maxpool(graph_com, out3_com)], dim=-1)
        scores = self.MLP_layer_global(hg3)
        scores_com = self.MLP_layer_global(hg3_com)
        return scores, scores_com, hg3_com, node_pred, node_score

    def loss(self, pred, label, hg_com):
        center = torch.mean(hg_com, dim=0).unsqueeze(0).repeat(hg_com.shape[0], 1)
        loss_com = self.mse(center, hg_com).item()
        if self.loss_type == 'cross':
            criterion = torch.nn.CrossEntropyLoss()
            loss_cls = criterion(pred, label.long())
        else:
            loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
        return loss_cls, loss_com
