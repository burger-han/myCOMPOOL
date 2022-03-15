import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math
import os
from matplotlib import cm

def acc_of_class(pred, label):

    #total acc
    scores = pred.detach().argmax(dim=1)
    acc = (scores == label).float().sum().item()

    #cls 0
    scores_cls0 = scores[(label==0).nonzero().squeeze(-1).long()]
    number0 = scores_cls0.size(0)
    acc_cls0 = (scores_cls0 == 0).float().sum().item() / number0

    #cls 1
    scores_cls1 = scores[(label==1).nonzero().squeeze(-1).long()]
    number1 = scores_cls1.size(0)
    acc_cls1 = (scores_cls1 == 1).float().sum().item() / number1

    return acc_cls0, acc_cls1, acc

def mse_distance(x):
    if type(x)==list:
        x= torch.cat(x,dim=0)
    x_centre = torch.mean(x, dim=0).unsqueeze(0).repeat(x.shape[0], 1)
    mse_loss = torch.nn.MSELoss()
    mseloss = mse_loss(x_centre, x)
    return mseloss

def vis_badcase(g_dist, g_com, y, corr_idx, bad_idx,vis_dir):
    corr_case = g_dist[corr_idx]
    bad_case = g_dist[bad_idx]
    y_badcase = y[bad_idx]
    #corr_case_com = g_com[corr_idx]
    bad_case_com = g_com[bad_idx]

    g_dist = g_dist.cpu().detach().numpy()
    g_com = g_com.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    bad_case = bad_case.cpu().detach().numpy()
    bad_case_com = bad_case_com.cpu().detach().numpy()
    y_badcase = y_badcase.cpu().detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13, 7)
# figure 1
    ax1.scatter(g_dist[:, 0], g_dist[:, 1], c=y, s=4)
    ax1.scatter(g_com[:, 0], g_com[:, 1], c='g', s=4)

# figure 2
    ax2.scatter(bad_case[:, 0], bad_case[:, 1], c=y_badcase, s=4)
    ax2.scatter(bad_case_com[:, 0], bad_case_com[:, 1], c='g', s=4)
    np.max(np.concatenate((g_dist, g_com), axis=0))
    max_range = np.max(np.concatenate((g_dist, g_com), axis=0))
    min_range = np.min(np.concatenate((g_dist, g_com), axis=0))

    x = np.linspace(min_range.item(), max_range.item(), 100)
    ax2.plot(x, x, '-r', label='class boundary')
    ax1.plot(x, x, '-r', label='class boundary')

    dir_list = vis_dir.split('/')
    vis_dir_new = os.path.join(dir_list[0], dir_list[1], 'vis_badcase', dir_list[3], vis_dir.split('/')[-1])

    plt.savefig(vis_dir_new)
    plt.close('all')
    return

def vis_position_and_scores( vis_dir, loss, acc, g_dist, g_com, y):
    #  趁着还是tensor 求一波 acc_of_class
    acc_cls0, acc_cls1, acc_all = acc_of_class(g_dist, y)
    '''
    fig1: 一个或者多个graph的信息： node score大小和node位置的关系
    fig2: node information
    '''
    # node_number = node_embedding.size(0)
    cls1_mse = mse_distance(g_dist[(y==1).nonzero().squeeze(-1).long()])
    cls0_mse = mse_distance(g_dist[(y==0).nonzero().squeeze(-1).long()])
    g_com_mse = mse_distance(g_com)

    g_dist = g_dist.cpu().detach().numpy()
    g_com = g_com.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    #
    # node_scores = node_scores.cpu().detach().numpy()
    # node_embedding = node_embedding.cpu().detach().numpy()
    # distance = (node_embedding[:, 1] - node_embedding[:, 0]) / math.sqrt(2)
    fig, (ax3, ax4) = plt.subplots(1, 2)
    fig.set_size_inches(12, 7)

# ax1.scatter(node_scores, distance, s=4)
#     ax1.set_xlabel('x of embedding', fontsize=12)
#     ax1.set_ylabel('y of node embedding', fontsize=12)
#     name1 = 'label:{}\n green: com\nblue: dist'.format(y[0]) + "\n node number: {}".format(node_number)
    # ax1.scatter(node_embedding[:,0], node_embedding[:,1], c=node_scores, s=10, cmap=plt.cm.hot_r)

    # ax1.scatter(g_dist[0][0], g_dist[0][1], c='blue',s=10)  # graph embedding
    # ax1.scatter(g_com[0][0], g_com[0][1], c='green',s=10)
    # 画一条y=x分割线
    # max_range1 = np.max(node_embedding)
    # min_range1 = np.min(node_embedding)
    #
    # x1= np.linspace(min_range1.item(), max_range1.item(), 100)
    # ax1.plot(x1, x1, '-r', label='class boundary')
    # ax1.set_title(name1)

# plt.colorbar(z1_plot, cax=ax1)
    #ax2.set_xlabel('scores', fontsize=12)
    #ax2.set_ylabel('number', fontsize=12)
    #ax2.hist(x=node_scores, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

    # name2 = 'node_score_distribution_(singlegraph)\n' + vis_dir.split('/')[-1]
    dir_list = vis_dir.split('/')
    vis_dir_new = os.path.join(dir_list[0], dir_list[1], dir_list[2], dir_list[3], vis_dir.split('/')[-1])
    # ax2.set_title(name2)

#   graph embedding distribution( distinct embedding and common embedding)
#     g_dist = g_dist.cpu().detach().numpy()
#     g_com = g_com.cpu().detach().numpy()
#     y = y.cpu().detach().numpy()
    number_of_nodes = g_dist.shape[0]
    # Scatter plot of data colored with labels
    # plt c 默认0对应的颜色是紫色，1对应黄色

    ax3.scatter(g_dist[:, 0], g_dist[:, 1], c=y, s=4)
    ax3.scatter(g_com[:, 0], g_com[:, 1], c='g', s=4)
    np.max(np.concatenate((g_dist, g_com), axis=0))
    max_range = np.max(np.concatenate((g_dist, g_com), axis=0))
    min_range = np.min(np.concatenate((g_dist, g_com), axis=0))

    x = np.linspace(min_range.item(), max_range.item(), 100)
    ax3.plot(x, x, '-r', label='class boundary')
    # get 几种类的center
    g_dist_class0 = g_dist[(y == 0).nonzero()]
    g_dist_class1 = g_dist[(y == 1).nonzero()]
    center_class0 = np.mean(g_dist_class0, axis=0)
    center_class1 = np.mean(g_dist_class1, axis=0)
    center_com = np.mean(g_com, axis=0)

#   cluster distribution
    distance0_ = (g_dist_class0[:, 1] - g_dist_class0[:, 0])/math.sqrt(2)
    distance1_ = (g_dist_class1[:, 1] - g_dist_class1[:, 0]) / math.sqrt(2)
    distance_com = (g_com[:, 1] - g_com[:, 0]) / math.sqrt(2)


    center0_dist = (center_class0[1] - center_class0[0]) / math.sqrt(2)
    center1_dist = (center_class1[1] - center_class1[0]) / math.sqrt(2)
    center_com_dist = (center_com[1] - center_com[0]) / math.sqrt(2)

    # ax4.axvline(center1_dist, linestyle='--', linewidth=2, color='yellow')
    # ax4.axvline(center0_dist, linestyle='--', linewidth=2, color='purple')
    # ax4.axvline(center_com_dist, linestyle='--', linewidth=2, color='green')
    # ax4.axvline(0, linestyle='--', linewidth=2, color='black')
    #
    # ax4.hist([distance0_, distance1_, distance_com], label=['class0', 'class1'])


    # 简易版distance分布
    # distance0 = g_dist_class0[:, 0] - center_class0[0]
    # distance_from_origin0 = 0 - center_class0[0]
    distance0_.sort()
    len0 = len(distance0_)
    ax4.barh(range(0, len0), distance0_, color='blue',edgecolor='none')
    ax4.axvline(center0_dist, linestyle='--', linewidth=2, color='purple')

    # distance1 = g_dist_class1[:, 0] - center_class1[0]
    # #print("distance1", distance1)
    # distance_from_origin1 = 0 - center_class1[0]
    distance1_.sort()
    len1 = len(distance1_)
    ax4.barh(range(len0, len0+len1), distance1_, color='orange', edgecolor='none')
    ax4.axvline(center1_dist, linestyle='--', linewidth=2, color='yellow')

    ax4.set_ylabel('number')
    ax4.set_title('acc cls0:{}, acc cls1:{}'.format("%.4f" % acc_cls0, "%.4f" % acc_cls1)+'\ndistance to center')

    center_common = np.mean(g_com, axis=0)
    ax3.scatter(center_class0[0], center_class0[1], marker='*', c='r', s=10)
    ax3.scatter(center_class1[0], center_class1[1], marker='*', c='orange', s=10)
    ax3.scatter(center_common[0], center_common[1], marker='*', c='b', s=10)
    name3= '\ntotal acc:{}  loss:{}'.format("%.4f" % acc, "%.4f" % loss) + \
          '\ncom_mse:{:.4f} cls0_mse:{:.4f} cls1_mse:{:.4f}'.format(g_com_mse.item(),cls0_mse.item(),cls1_mse.item())
    ax3.set_title(name3)
    plt.savefig(vis_dir_new)
    plt.close('all')

    return

def plot_g_com_dist(g_dist, g_com, y, vis_dir):
    '''
    input：g_dist：useful node组成的 graph embedding
           com_dist：useful node组成的 graph embedding
    '''
    g_dist = g_dist.cpu().detach().numpy()
    g_com = g_com.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    number_of_nodes = g_dist.shape[0]
    fig, ax1 = plt.subplots()
    # Scatter plot of data colored with labels

# plt c 默认0对应的颜色是紫色，1对应黄色
    ax1.scatter(g_dist[:, 0], g_dist[:, 1], c=y, s=4)
    ax1.scatter(g_com[:, 0], g_com[:, 1], c='g', s=4)
    np.max(np.concatenate((g_dist, g_com), axis=0))
    max_range = np.max(np.concatenate((g_dist, g_com), axis=0))
    min_range = np.min(np.concatenate((g_dist, g_com), axis=0))

    x = np.linspace(min_range.item(), max_range.item(),100)
    ax1.plot(x, x, '-r', label='class boundary')
# get 几种类的center
    g_dist_class0 = g_dist[(y==0).nonzero()]
    g_dist_class1 = g_dist[y]
    center_class0 = np.mean(g_dist_class0, axis=0)
    center_class1 = np.mean(g_dist_class1, axis=0)

    center_common = np.mean(g_com, axis=0)
    ax1.scatter(center_class0[0], center_class0[1], marker='*', c='r', s=10)
    ax1.scatter(center_class1[0], center_class1[1], marker='*', c='orange', s=10)
    ax1.scatter(center_common[0], center_common[1], marker='*', c='b', s=10)

    name = vis_dir.split('/')[-1]
    title = name + "center0:red, center1:orange, common center:blue"
    plt.title(title)
    plt.savefig(vis_dir)
    # plt.plot(x, y)
    plt.close('all')
    return


def plot_scatter(g, y, vis_dir):
    g = g.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    plt.scatter(g[:, 0], g[:, 1], c=y, s=10, cmap=plt.cm.Paired)

    x_min = min(g[:, 0].min() - 0.5, g[:, 1].min() - 0.5)
    x_max = max(g[:, 0].max() + 0.5, g[:, 1].max() + 0.5)

    x_range = np.arange(x_min, x_max)
    y_range = x_range
    plt.plot(x_range, y_range)

    name = vis_dir.split('/')[-1]
    title = name
    plt.title(title)
    plt.savefig(vis_dir)
    plt.close('all')
    return

def vis(xs, labels, vis_dir, acc, loss):
    X_std = torch.cat(xs).cpu().numpy()
    label = torch.cat(labels).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Scatter plot of data colored with labels
    ax2.scatter(X_std[:, 0], X_std[:, 1], c=label)
    x_min = min(X_std[:, 0].min(), X_std[:, 1].min())
    x_max = max(X_std[:, 0].max(), X_std[:, 1].max())
    x = np.arange(x_min, x_max)
    y = x
    plt.plot(x, y)

    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()

    name = vis_dir.split('/')[-1]
    title = '{}  acc:{}, loss:{}'.format(name, '%.2f' % acc, '%.2f' % loss,)
    plt.title(title)
    plt.savefig(vis_dir)
    plt.close('all')



def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep

def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def global_global_loss(g_useful, g_useless, measure, device, args):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    if args.contras_v == 1: # positive: useless nodes, negative: useful nodes
        num_graphs = g_useful.shape[0]
        pos_mask = torch.ones((num_graphs, num_graphs)).to(device)
        neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
        for graphidx in range(num_graphs):
            pos_mask[graphidx][graphidx] = 0.
            neg_mask[graphidx][graphidx] = 0.

        res_useful = torch.mm(g_useful, g_useful.t())
        res_useless = torch.mm(g_useless, g_useless.t())
        E_pos = get_positive_expectation(res_useless * pos_mask, measure, average=False).sum()
        E_pos = E_pos / num_graphs
        E_neg = get_negative_expectation(res_useful * neg_mask, measure, average=False).sum()
        E_neg = E_neg / (num_graphs * (num_graphs - 1))

    elif args.contras_v == 2: # positive: useless nodes, negative: 全局-positive
        g_total = torch.cat((g_useful, g_useless), dim=0)
        num_graphs = g_useful.shape[0] * 2
        pos_mask = torch.zeros((num_graphs, num_graphs)).to(device)
        neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
        for graphidx in range(g_useful.shape[0]):
            pos_mask[graphidx][graphidx] = 1.
        for idx1 in range(g_useful.shape[0]):
            for idx2 in range(g_useful.shape[0]):
                neg_mask[idx1][idx2] = 0.
        for idx in range(g_useful.shape[0]*2):
            neg_mask[idx][idx] = 0.

        res = torch.mm(g_total, g_total.t())

        E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
        E_pos = E_pos / num_graphs
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
        E_neg = E_neg / (num_graphs * (num_graphs - 1))

    return E_neg - E_pos

