import torch
from utils import augment_data
from loss import *
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


def get_graph_embedding(data, model):
    model.eval()
    with torch.no_grad():
        z, _ = model(data, data)

    return z