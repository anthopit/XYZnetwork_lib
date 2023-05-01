from torch_geometric.utils import from_networkx
from torch_geometric.utils import negative_sampling
from GNN.utils import *
import numpy as np
def create_data_from_transport_network(graph, TN, *args, **kwargs):
    """
    Create train and validation data from a transport network graph.

    This function takes a networkx graph representation of a transport network and a `GNNConfig` object, which specifies the parameters for generating the node features and dividing the data into train and validation sets. The function then generates node features for the graph based on the specified features in the `GNNConfig` object, divides the data into train and validation sets according to the specified ratios, and returns a `Data` object from PyTorch Geometric containing the graph data and the train and validation masks.

    Parameters
    ----------
    graph : networkx graph
        A networkx graph representing the transport network.
    TN : dict
        A dictionary containing the transport network data.
    *args, **kwargs : optional
        Optional arguments and keyword arguments for the `GNNConfig` object.

    Returns
    -------
    data : PyTorch Geometric `Data` object
        The generated graph data and train/validation masks.
    """
    if args and isinstance(args[0], GNNConfig):
        config = args[0]
    else:
        config = GNNConfig(*args, **kwargs)

    # Create a graph from the transport network
    data = from_networkx(graph, group_edge_attrs=config.edge_attrs, group_node_attrs=config.node_attrs)

    # Add the differents node features
    for feature in config.node_features:
        data = cat_node_feature(data, graph, TN, feature=feature, num_workers=config.num_workers)

    # Divide the data into train and validation
    num_nodes = data.num_nodes
    num_train_nodes = int(num_nodes * config.train_ratio)
    num_val_nodes = int(num_nodes * config.val_ratio)

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


def create_link_prediction_data(data, train_ratio=0.8):
    """
    Create train and validation data for link prediction.

    This function takes a PyTorch Geometric data object representing a graph, generates positive examples from existing edges, and negative examples using negative sampling. The data is then shuffled and split into training and testing sets according to the specified ratio.

    Parameters
    ----------
    data : PyTorch Geometric Data object
        A graph data object with attributes 'num_nodes', 'num_edges', and 'edge_index'.
    train_ratio : float, optional
        The ratio of the dataset to be used for training, defaults to 0.8.

    Returns
    -------
    train_examples : numpy array
        The array of training examples.
    train_labels : numpy array
        The array of training labels.
    test_examples : numpy array
        The array of testing examples.
    test_labels : numpy array
        The array of testing labels.

    Example
    -------
    >>> train_examples, train_labels, test_examples, test_labels = create_link_prediction_data(data, train_ratio=0.8)
    """
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