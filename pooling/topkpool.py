import torch
import matplotlib.pyplot as plt
from torch_scatter import scatter_add, scatter_max
from ultils import *
import logging
from scipy.stats import t
import math


def get_batch_id(num_nodes:torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def TopKPooling(x:torch.Tensor, ratio, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.
    
    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    ratio = float(ratio)
    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)]
    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    return perm, k

def get_dense_x(batch, x):
    #print("1x :{}, 2batch:{}".format(x.shape, batch.shape))
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)
    dense_x = x.new_full((batch_size * max_num_nodes,),
                         torch.finfo(x.dtype).min)
    if x.dim() == 2:
        x = x.squeeze(-1)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)
    return dense_x, num_nodes, max_num_nodes, cum_num_nodes

def mytopk(x, smallratio, ratio, batch):
    # 已经修改成了输出的perm都是distinct（useful）node的
    if x.dim() == 2:
        x = x.squeeze(-1)
    #普通版本topk 对每个graph取fixed number或者ratio 的node，需要构造一个dense graph
    dense_x, num_nodes, max_num_nodes, cum_num_nodes = get_dense_x(batch, x)
    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    if ratio > 1: # 此时ratio表示useless数量
        k = num_nodes.new_full((num_nodes.size(0), ), ratio)
        for idx in range(len(k)):   # 小graph 不够pool数，保留整个graph
            if k[idx] < num_nodes[idx]:
                k[idx] = int(k[idx])
            else:
                if int(smallratio) == 1:
                    k[idx] = 0
                else:
                    k[idx] = smallratio * num_nodes[idx]
        k = torch.sub(num_nodes, k)
    else:
        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [torch.arange(k[i], dtype=torch.long, device=x.device) + i * max_num_nodes for i in range(k.size(0))]
    mask = torch.cat(mask, dim=0)
    perm_new = perm[mask]
    return perm_new, k


def vis_pooling(x, perm,vis_dir):
    label = torch.zeros(x.size(0))
    label[perm]=1
    plt.scatter(x[:, 0], x[:, 1], c=label, s=15, cmap=plt.cm.Paired)
    name = vis_dir.split('/')[-1]
    plt.title(name)
    plt.savefig(vis_dir)
    # plt.clf()
    plt.close('all')
    return
