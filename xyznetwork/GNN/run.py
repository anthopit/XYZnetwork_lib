import torch
from GNN.utils import augment_data
from GNN.loss import *
def train_self_supervised(data, model, optimizer, args):
    """
    Train a self-supervised GNN model using contrastive learning.
    This function trains a GNN model using self-supervised contrastive learning with the InfoNCE loss. The training is performed on the train set and the model's performance is evaluated on the validation set.
    Parameters
    ----------
    data : torch_geometric.data.Data
        The input graph data, represented as a PyTorch Geometric Data object.
    model : torch.nn.Module
        The GNN model that takes graph data as input and produces embeddings.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the self-supervised GNN model.
    args : object
        An object containing the necessary arguments for training, such as the number of epochs, loss type, and data augmentation methods.
    Returns
    -------
    None
    Example
    -------
    >>> # Create data
    >>> data = create_data_from_transport_network(TN.graph, TN, args).to(args.device)
    >>> # Create model
    >>> ssl_model = SSL_GNN(data.num_node_features, args).to(args.device)
    >>> # Create the optimizer
    >>> optimizer = torch.optim.Adam(ssl_model.parameters(), lr=args.lr)
    >>> train_self_supervised(graph_data, gnn_model, optimizer, args)
    """

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Augment data
        data_train = data.subgraph(data.train_mask)
        data_aug = augment_data(data_train, augment_list=args.augment_list)

        # Forward pass with the original and augmented data
        z1_train, z2_train = model(data_train, data_aug.to(args.device))

        # Compute InfoNCE loss for training set
        train_loss = None
        if args.loss == "infonce":
            train_loss = info_nce_loss(z1_train, z2_train)

        # Backpropagation
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Augment data
            data_val = data.subgraph(data.val_mask)
            data_val_aug = augment_data(data_val)

            # Forward pass with the original and augmented data
            z1_val, z2_val = model(data_val, data_val_aug.to(args.device))

            # Compute InfoNCE loss for validation set
            val_loss = info_nce_loss(z1_val, z2_val)

        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

        # Save the best model weights based on validation loss
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), args.save)

    print(f'Best Val Loss: {best_loss}')

def train_supervised(data, ssl_model, optimizer, args):
    """
    Train a supervised GNN model using node embeddings generated by a self-supervised learning model.
    This function trains a GNN model for a supervised task using the node embeddings obtained from a pre-trained self-supervised learning (SSL) model. The training is performed on the train set and the model's performance is evaluated on the validation set.
    Parameters
    ----------
    data : torch_geometric.data.Data
        The input graph data, represented as a PyTorch Geometric Data object.
    ssl_model : torch.nn.Module
        A pre-trained self-supervised learning model that takes graph data as input and produces embeddings.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the supervised GNN model.
    args : object
        An object containing the necessary arguments for training, such as the number of epochs.
    Returns
    -------
    None
    """
    criterion = torch.nn.MSELoss()

    train_data = data.subgraph(data.train_mask)
    val_data = data.subgraph(data.val_mask)

    # Training loop
    for epoch in range(400):
        train_loss = train(train_data)
        val_loss = validate(val_data)
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

def train(train_data, model, optimizer, criterion, args):
    model.train()
    loss_all = 0
    train_data = train_data.to(args.device)
    optimizer.zero_grad()
    output = model(train_data.x, train_data.edge_index)
    loss = criterion(output[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    loss_all += loss.item()
    return loss_all / len(train_data.y)

def validate(val_data, model, criterion, args):
    model.eval()
    loss_all = 0
    val_data = val_data.to(args.device)
    output = model(val_data.x, val_data.edge_index)
    loss = criterion(output[val_data.val_mask], val_data.y[val_data.val_mask])
    loss_all += loss.item()
    return loss_all / len(val_data.y)

def get_graph_embedding(data, model):
    """
    Obtain graph embeddings using a trained GNN model.
    This function computes the node embeddings for a given graph by passing its data through a trained Graph Neural Network (GNN) model. The embeddings can be used for downstream tasks such as link prediction, node classification, or graph clustering.
    Parameters
    ----------
    data : torch_geometric.data.Data
        The input graph data, represented as a PyTorch Geometric Data object.
    model : torch.nn.Module
        A trained GNN model that takes graph data as input and produces embeddings.
    Returns
    -------
    z : torch.Tensor
        The computed node embeddings for the input graph.
    """
    model.eval()
    with torch.no_grad():
        z, _ = model(data, data)

    return