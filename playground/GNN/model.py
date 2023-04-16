import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GINConv
from torch.nn import Sequential, Linear, ReLU
import torch

class GCNLinkPrediction(torch.nn.Module):
    def __init__(self, model, grad=True):
        super(GCNLinkPrediction, self).__init__()

        # Use the GNN model
        self.model = model
        self.grad = grad

        # Set requires_grad to True for pretrained layers
        for param in self.model.parameters():
            param.requires_grad = grad

        if isinstance(model, SSL_GNN):
            gnn_model = model.gnn
        else:
            gnn_model = model

        if gnn_model.output_layer is not None:
            out_features = gnn_model.output_layer.out_channels
        else:
            out_features = gnn_model.hidden_layers[-1].out_channels

        # Add a bilinear layer for link prediction
        self.link_pred_layer = torch.nn.Bilinear(gnn_model.fc[-1].out_features, gnn_model.fc[-1].out_features, 1)

    def forward(self, x, edge_index):
        # Get node embeddings from the GNN model
        if isinstance(self.model, SSL_GNN):
            node_embeddings = self.model.gnn(x, edge_index)
        else:
            node_embeddings = self.model(x, edge_index)

        return node_embeddings

    def predict_link(self, node_embeddings, edge_list):
        edge_embeddings1 = node_embeddings[edge_list[0]]
        edge_embeddings2 = node_embeddings[edge_list[1]]
        scores = self.link_pred_layer(edge_embeddings1, edge_embeddings2)
        return scores

class SSL_GNN(torch.nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim, model='gcn'):
        super(SSL_GNN, self).__init__()
        self.model = model
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        if model == 'gcn':
            self.gnn = GCN(feat_dim, num_layers, hidden_dim, embed_dim)
        elif model == 'sage':
            self.gnn = GraphSAGE(feat_dim, num_layers, hidden_dim, embed_dim)
        elif model == 'gat':
            self.gnn = GAT(feat_dim, num_layers, hidden_dim, embed_dim, in_heads=1, out_heads=1)
        elif model == 'gin':
            self.gnn = GIN(feat_dim, num_layers, hidden_dim, embed_dim)

    def forward(self, data, data_aug):
        x, edge_index = data.x, data.edge_index
        x_aug, edge_index_aug = data_aug.x, data_aug.edge_index

        return self.gnn(x, edge_index), self.gnn(x_aug, edge_index_aug)


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim):
        super(GCN, self).__init__()

        # Input layer
        self.input_layer = GCNConv(feat_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            self.output_layer = GCNConv(hidden_dim, embed_dim)
        else:
            self.output_layer = None

        # Fully connected layers for the embedding
        if num_layers > 1:
            self.fc = Sequential(Linear(embed_dim, hidden_dim), ReLU(), Linear(hidden_dim, embed_dim))
        else:
            self.fc = torch.nn.Linear(hidden_dim, embed_dim)

        # Xavier uniform initialization for linear layers
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.input_layer(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        if self.output_layer is not None:
            x = F.relu(self.output_layer(x, edge_index))
        x = self.fc(x)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim):
        super(GraphSAGE, self).__init__()

        # Input layer
        self.input_layer = SAGEConv(feat_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            self.output_layer = SAGEConv(hidden_dim, embed_dim)
        else:
            self.output_layer = None

        # Fully connected layers for the embedding
        if num_layers > 1:
            self.fc = Sequential(Linear(embed_dim, hidden_dim), ReLU(), Linear(hidden_dim, embed_dim))
        else:
            self.fc = torch.nn.Linear(hidden_dim, embed_dim)

        # Xavier uniform initialization for linear layers
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.input_layer(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        if self.output_layer is not None:
            x = F.relu(self.output_layer(x, edge_index))
        x = self.fc(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim, in_heads, out_heads):
        super(GAT, self).__init__()

        # Input layer
        self.input_layer = GATConv(feat_dim, hidden_dim, heads=in_heads)

        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATConv(hidden_dim * in_heads, hidden_dim, heads=in_heads))

        # Output layer
        if num_layers > 1:
            self.output_layer = GATConv(hidden_dim * in_heads, embed_dim // out_heads, heads=out_heads)
        else:
            self.output_layer = None

        # Fully connected layers for the embedding
        if num_layers > 1:
            self.fc = Sequential(Linear(embed_dim, hidden_dim), ReLU(), Linear(hidden_dim, embed_dim))
        else:
            self.fc = torch.nn.Linear(hidden_dim, embed_dim)

        # Xavier uniform initialization for linear layers
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.input_layer(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        if self.output_layer is not None:
            x = F.relu(self.output_layer(x, edge_index))
        x = self.fc(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = Linear(input_dim, output_dim)
        self.linear2 = Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim):
        super(GIN, self).__init__()

        # Input layer
        self.input_layer = GINConv(MLP(feat_dim, hidden_dim))

        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GINConv(MLP(hidden_dim, hidden_dim)))

        # Output layer
        if num_layers > 1:
            self.output_layer = GINConv(MLP(hidden_dim, embed_dim))
        else:
            self.output_layer = None

        # Fully connected layers for the embedding
        if num_layers > 1:
            self.fc = Sequential(Linear(embed_dim, hidden_dim), ReLU(), Linear(hidden_dim, embed_dim))
        else:
            self.fc = torch.nn.Linear(hidden_dim, embed_dim)

        # Xavier uniform initialization for linear layers
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.input_layer(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        if self.output_layer is not None:
            x = F.relu(self.output_layer(x, edge_index))
        x = self.fc(x)

        return x