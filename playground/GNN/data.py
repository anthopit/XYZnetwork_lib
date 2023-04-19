from torch_geometric.utils import from_networkx
from torch_geometric.utils import negative_sampling
from utils import *
import numpy as np
def create_data_from_transport_network(graph, TN,
                                       node_features=['degree_one_hot'],
                                       node_attrs=None, edge_attrs=None,
                                       node_label=None,
                                       train_ratio = 0.8, val_ratio = 0.2,
                                       num_workers=1):

    # Create a graph from the transport network
    data = from_networkx(graph, group_edge_attrs=edge_attrs, group_node_attrs=node_attrs)

    # Add the differents node features
    for feature in node_features:
        data = cat_node_feature(data, graph, TN, feature=feature, num_workers=num_workers)

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

# Prepare the dataset for link prediction
def create_link_prediction_data(data, train_ratio=0.8):
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    # Create positive examples
    edge_index = data.edge_index
    pos_examples = edge_index.t().cpu().numpy()

    # Create negative examples
    neg_examples = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_edges).t().cpu().numpy()

    # Combine positive and negative examples
    all_examples = np.vstack([pos_examples, neg_examples])
    labels = np.hstack([np.ones(len(pos_examples)), np.zeros(len(neg_examples))])

    # Shuffle and split the dataset
    indices = np.random.permutation(len(all_examples))
    train_size = int(train_ratio * len(indices))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_examples = all_examples[train_indices]
    train_labels = labels[train_indices]

    test_examples = all_examples[test_indices]
    test_labels = labels[test_indices]

    return train_examples, train_labels, test_examples, test_labels