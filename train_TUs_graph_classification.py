"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
from sklearn.metrics import silhouette_samples, silhouette_score
from metrics import accuracy_TU as accuracy
from pooling.topkpool import get_batch_id
from ultils import plot_g_com_dist
from ultils import *

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch, vis_dir):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    epoch_train_acc_full = 0
    nb_data = 0
    gpu_mem = 0
    batch_id_epoch, node_scores_epoch, node_pred_epoch = [], [], []
    epoch_scores, epoch_scores_com, epoch_labels = [], [], []
    epoch_mse0, epoch_mse1 = [], []
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_id_epoch.append(get_batch_id(batch_graphs.batch_num_nodes()))
        if model.name == 'COMPool':
            batch_scores, batch_scores_com, hg, node_pred, node_scores = model.forward(batch_graphs, batch_x, batch_labels, batch_e)
            loss_cls, loss_com = model.loss(batch_scores, batch_labels, hg)
            # cls0_mse_batch = mse_distance(batch_scores[(batch_labels == 0).nonzero().squeeze(-1).long()])
            # cls1_mse_batch = mse_distance(batch_scores[(batch_labels == 1).nonzero().squeeze(-1).long()])
            model_k = model.k
            # loss = loss_cls + model_k * loss_com/ (cls0_mse_batch + cls1_mse_batch)
            #loss = loss_cls + model_k * loss_com
            loss = loss_cls
            epoch_scores.append(batch_scores)
            epoch_scores_com.append(batch_scores_com)
            node_scores_epoch.append(node_scores)
            node_pred_epoch.append(node_pred)
        elif model.name == 'SAGPool':
            batch_scores, batch_scores_com, batch_scores_full, node_pred, node_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
            epoch_scores.append(batch_scores)
            epoch_scores_com.append(batch_scores_com)
            node_scores_epoch.append(node_scores)
            node_pred_epoch.append(node_pred)
        elif model.name =='SAGPool_b':
            print("model.name == SAGPool_b")
            batch_scores, mse_loss = model.forward(batch_graphs, batch_x, batch_e)
            #loss = model.loss(batch_scores, batch_labels) + mse_loss
            loss = model.loss(batch_scores, batch_labels)
        elif model.name == 'Mewis':
            batch_scores, loss_pool = model(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels, loss_pool)
        elif model.name == 'HGPSLPool':
            batch_scores = model(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
        elif model.name == 'NodeLevel':
            x, x_g, g, node_label= model.forward(batch_graphs, batch_labels, batch_x, batch_e)
            loss = model.loss(g, x_g, batch_labels, node_label)
            batch_scores = g
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
        # plot_scatter(batch_scores, batch_labels, vis_dir)
        epoch_labels.append(batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    if model.name == 'COMPool' or 'SAGPool':
        g_dist = torch.cat(tuple(epoch_scores), dim=0)
        g_com = torch.cat(tuple(epoch_scores_com), dim=0)
        y = torch.cat(tuple(epoch_labels), dim=0)
        mse_cls1 = mse_distance(g_dist[(y == 1).nonzero().squeeze(-1).long()])
        mse_cls0 = mse_distance(g_dist[(y == 0).nonzero().squeeze(-1).long()])
        mse_com = mse_distance(g_com)
        epoch_train_acc_full /= nb_data
        corr_idx = (torch.argmax(g_dist, dim=-1) == y).nonzero().squeeze(-1).long()
        bad_idx = (torch.argmax(g_dist, dim=-1) != y).nonzero().squeeze(-1).long()

#
#     if epoch % 50 == 0 and vis_dir is not None:
#         if model.name == 'COMPool' or 'SAGPool':
#             batch_id_epoch = torch.cat(tuple(batch_id_epoch),dim=0)
# #            vis_badcase(g_dist, g_com, y, corr_idx, bad_idx,vis_dir)
#             vis_position_and_scores(vis_dir,
#                                     epoch_loss, epoch_train_acc, g_dist, g_com, y)
    if model.name == 'COMPool' or 'SAGPool':
        return epoch_loss, epoch_train_acc, optimizer, mse_cls1, mse_cls0, mse_com
    else:
        return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch, vis_dir=None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_acc_full = 0
    nb_data = 0
    batch_id_epoch, node_scores_epoch, node_pred_epoch = [],[],[]
    epoch_scores, epoch_scores_com, epoch_labels = [],[],[]
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_id_epoch.append(get_batch_id(batch_graphs.batch_num_nodes()))
            if model.name == 'COMPool':
                batch_scores, batch_scores_com,  hg_com, node_pred, node_scores = model.forward(batch_graphs, batch_x, batch_labels, batch_e)
                loss_cls, loss_com = model.loss(batch_scores, batch_labels,  hg_com)
                cls0_mse_batch = mse_distance(batch_scores[(batch_labels==0).nonzero().squeeze(-1).long()])
                cls1_mse_batch = mse_distance(batch_scores[(batch_labels==1).nonzero().squeeze(-1).long()])
                model_k = model.k
                # loss = loss_cls + model_k * loss_com / (cls0_mse_batch+cls1_mse_batch)
                #loss = loss_cls + model_k * loss_com
                loss = loss_cls
                # batch append to epoch
                epoch_scores.append(batch_scores)
                epoch_scores_com.append(batch_scores_com)
                node_scores_epoch.append(node_scores)
                node_pred_epoch.append(node_pred)

            elif model.name == 'SAGPool':
                batch_scores, batch_scores_com, batch_scores_full, node_pred, node_scores = model.forward(
                        batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels)
                epoch_scores.append(batch_scores)
                epoch_scores_com.append(batch_scores_com)
                node_scores_epoch.append(node_scores)
                node_pred_epoch.append(node_pred)
            elif model.name =='SAGPool_b':
                batch_scores, mse_loss = model.forward(batch_graphs, batch_x, batch_e)
                #loss = model.loss(batch_scores, batch_labels) + mse_loss
                loss = model.loss(batch_scores, batch_labels)

            elif model.name == 'Mewis':
                batch_scores, loss_pool = model(batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels, loss_pool)

            elif model.name == 'HGPSLPool':
                batch_scores = model(batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels)

            elif model.name == 'NodeLevel':
                x, x_g, g, node_label = model.forward(batch_graphs, batch_labels, batch_x, batch_e)
                loss = model.loss(g, x_g, batch_labels, node_label)
                batch_scores = g
            else:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels)
            epoch_labels.append(batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
            epoch_scores.append(batch_scores)
            epoch_labels.append(batch_labels)
        if model.name == 'COMPool' or 'SAGPool':
            g_dist = torch.cat(tuple(epoch_scores), dim=0)
            g_com = torch.cat(tuple(epoch_scores_com), dim=0)
            y = torch.cat(tuple(epoch_labels), dim=0)
            mse_cls1 = mse_distance(g_dist[(y == 1).nonzero().squeeze(-1).long()])
            mse_cls0 = mse_distance(g_dist[(y == 0).nonzero().squeeze(-1).long()])
            mse_com = mse_distance(g_com)
            epoch_acc_full /= nb_data

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        if epoch % 50 == 0 and vis_dir is not None:
            if model.name == 'COMPool' or 'SAGPool':
                #batch_id_epoch = torch.cat(tuple(batch_id_epoch), dim=0)
                vis_position_and_scores(vis_dir, epoch_test_loss, epoch_test_acc,
                                        torch.cat(tuple(epoch_scores), dim=0),
                                        torch.cat(tuple(epoch_scores_com), dim=0),
                                        torch.cat(tuple(epoch_labels), dim=0))

    if model.name == 'COMPool' or 'SAGPool':
        return epoch_test_loss, epoch_test_acc, mse_cls1, mse_cls0, mse_com
    else:
        return epoch_test_loss, epoch_test_acc

"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size,vis_dir=None):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_dense(model, device, data_loader, epoch, vis_dir=None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc

def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter

def eval_cluster(coor, label, eps=1.e-6):
    same_num = 0
    HM_sum = 0
    size = coor.shape[0]
    node_dis = np.zeros([size, size])
    for i in range(size):
        node_dis[i][i] = float('inf')
        for j in range(i + 1, size):
            node_dis[i][j] = node_dis[j][i] = np.linalg.norm(coor[i] - coor[j]) + eps
        if label[i] == label[node_dis[i].argmin()]:
            same_num += 1
        same_idx, diff_index = (label == label[i]).nonzero(), (label != label[i]).nonzero()
        HM_sum += node_dis[i][diff_index].min() / node_dis[i][same_idx].min()
    return same_num / size, HM_sum / size
