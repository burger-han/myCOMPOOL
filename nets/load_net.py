"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
from nets.gin_net import GINNet
from nets.mo_net import MoNet as MoNet_
from nets.mlp_net import MLPNet
from nets.ring_gnn_net import RingGNNNet
from nets.three_wl_gnn_net import ThreeWLGNNNet
from nets.compool import *
from pooling.sagpool import *
from pooling.sagpool_baseline import *
from pooling.mewispool import *
from pooling.hgpslpool import *
from nets.node_graph_net import *

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def RingGNN(net_params):
    return RingGNNNet(net_params)

def ThreeWLGNN(net_params):
    return ThreeWLGNNNet(net_params)

def COMPool(net_params):
    return COMPoolNet(net_params)

def SAGPoolNet(net_params):
    return SAGPoolReadout(net_params)

def SAGPoolNet_b(net_params):
    return SAGPoolReadout_b(net_params)

def NodeGraph(net_params):
    return NodeGraphNet(net_params)

def MEWISPool(net_params):
    return MEWISPoolNet(net_params)

def HGPSLPool(net_params):
    return HGPSLPoolReadout(net_params)



def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet_,
        'MLP': MLP,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN,
        'COMPool': COMPool,
        'SAGPool': SAGPoolNet,
        'SAGPool_b': SAGPoolNet_b,
        'NodeLevel': NodeGraph,
        'Mewis': MEWISPool,
        'HGPSLPool': HGPSLPoolReadout
    }
    model = models[MODEL_NAME](net_params)
    model.sag_type = net_params["sag_type"]
    if model.sag_type == 'o':
        model = COMPoolNet(net_params)
    elif model.sag_type == 'gumble':
        model = COMPoolNet_gumble(net_params)
    elif model.sag_type =='global':
        model = COMPoolNet_global(net_params)
    elif model.sag_type == 'h':
        model = COMPoolNet_h(net_params)
    model.name = MODEL_NAME
    model.topk = net_params["topk"]
    model.k = net_params["k"]

    return model