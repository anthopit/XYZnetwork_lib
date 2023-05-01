from GNN.loss import *
from GNN.utils import augment_data
import torch
def train_self_supervised(data, model, optimizer, args):

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

def train_self_supervised(data, ssl_model, optimizer, args):
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
    Gives the graph embedding from a model.
    :param data: Data to use
    :param model: Model to use
    :return: Graph embedding
    """
    model.eval()
    with torch.no_grad():
        z, _ = model(data, data)

    return z