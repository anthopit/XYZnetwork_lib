from torch_geometric.utils import from_networkx
from utils import *
import numpy as np
def create_data_from_transport_network(TN,
                                       node_features=['degree_one_hot'],
                                       node_attrs=None, edge_attrs=None,
                                       node_label=None,
                                       train_ratio = 0.8, val_ratio = 0.2,
                                       num_workers=1):

    # Create a graph from the transport network
    data = from_networkx(TN.get_higher_complexity(), group_edge_attrs=edge_attrs, group_node_attrs=node_attrs)

    # Add the differents node features
    for feature in node_features:
        data = cat_node_feature(data, TN, feature=feature, num_workers=num_workers)

    # Divide the data into train and validation
    num_nodes = data.num_nodes
    num_train_nodes = int(num_nodes * train_ratio)
    num_val_nodes = int(num_nodes * val_ratio)

    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_indices = indices[:num_train_nodes]
    val_indices = indices[num_train_nodes:num_train_nodes + num_val_nodes]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask

    return data