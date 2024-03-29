import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GINConv
from torch.nn import Sequential, Linear, ReLU
import torch
from GNN.utils import GNNConfig

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
        """
        Forwards node embeddings
        :param x: Node
        :param edge_index: Edge index
        :return: Node embeddings
        """
        # Get node embeddings from the GNN model
        if isinstance(self.model, SSL_GNN):
            node_embeddings = self.model.gnn(x, edge_index)
        else:
            node_embeddings = self.model(x, edge_index)

        return node_embeddings

    def predict_link(self, node_embeddings, edge_list):
        """
        Predicts links from node embeddings
        :param node_embeddings: Node embeddings to use
        :param edge_list: List of edges
        :return: Prediction scores
        """
        edge_embeddings1 = node_embeddings[edge_list[0]]
        edge_embeddings2 = node_embeddings[edge_list[1]]
        scores = self.link_pred_layer(edge_embeddings1, edge_embeddings2)
        return scores

class SSL_GNN(torch.nn.Module):
    """
    Initializes the SSL_GNN class, which is a self-supervised learning model on top of a classic graph neural
    network (GNN) model.
    Parameters
    ----------
    feat_dim : int
        The dimensionality of the input node features.
    *args : list
        Variable length argument list. If the first argument is an instance of `GNNConfig`, the configuration is
        taken from that instance, otherwise the arguments are passed to `GNNConfig` to create a new configuration.
    **kwargs : dict
        Keyword arguments that are passed to `GNNConfig` to create a new configuration.
    Attributes
    ----------
    feat_dim : int
        The dimensionality of the input node features.
    model : str
        The type of GNN to use.
    hidden_dim : int
        The dimensionality of the hidden layers of the GNN.
    embed_dim : int
        The dimensionality of the output node embeddings.
    gnn : torch.nn.Module
        The GNN module to use.
    Returns
    -------
    None
    """
    def __init__(self, feat_dim, *args, **kwargs):
        super(SSL_GNN, self).__init__()

        if args and isinstance(args[0], GNNConfig):
            config = args[0]
        else:
            config = GNNConfig(*args, **kwargs)

        self.feat_dim = feat_dim
        self.model = config.model
        self.hidden_dim = config.hidden_dim
        self.embed_dim = config.dim_embedding
        if config.model == 'gcn':
            self.gnn = GCN(feat_dim, config.layers, config.hidden_dim, config.dim_embedding)
        elif config.model == 'sage':
            self.gnn = GraphSAGE(feat_dim, config.layers, config.hidden_dim, config.dim_embedding)
        elif config.model == 'gat':
            self.gnn = GAT(feat_dim, config.layers, config.hidden_dim, config.dim_embedding, in_heads=1,
                               out_heads=1)
        elif config.model == 'gin':
            self.gnn = GIN(feat_dim, config.layers, config.hidden_dim, config.dim_embedding)

    def forward(self, data, data_aug):
        """
        Forward propagation of the SSL_GNN class.
        Parameters
        ----------
        data : torch_geometric.data.Data
            The original data to use.
        data_aug : torch_geometric.data.Data
            The augmented data to use.
        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            The embeddings of the original data and the augmented data.
        """
        x, edge_index = data.x, data.edge_index
        x_aug, edge_index_aug = data_aug.x, data_aug.edge_index

        return self.gnn(x, edge_index), self.gnn(x_aug, edge_index_aug)


class GCN(torch.nn.Module):
    """
    A graph convolutional neural network (GCN) used for graph embedding.
    Inherits from the torch.nn.Module class.
    Parameters
    ----------
    feat_dim : int
        The dimensionality of the input features.
    num_layers : int
        The number of hidden layers in the GCN.
    hidden_dim : int
        The dimensionality of the hidden layers.
    embed_dim : int
        The dimensionality of the output embedding.
    Attributes
    ----------
    input_layer : torch_geometric.nn.conv.GCNConv
        The input layer of the GCN.
    hidden_layers : torch.nn.ModuleList
        A list of hidden layers in the GCN.
    output_layer : torch_geometric.nn.conv.GCNConv or None
        The output layer of the GCN, or None if num_layers is 1.
    fc : torch.nn.Sequential or torch.nn.Linear
        The fully connected layers used to compute the embedding.
    Methods
    -------
    forward(x, edge_index)
        Computes the forward pass of the GCN.
    """
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim):
        """
        Initializes a new instance of the GCN class.
        Parameters
        ----------
        feat_dim : int
            The dimensionality of the input features.
        num_layers : int
            The number of hidden layers in the GCN.
        hidden_dim : int
            The dimensionality of the hidden layers.
        embed_dim : int
            The dimensionality of the output embedding.
        """
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
        """
        Computes the forward pass of the GCN.
        Parameters
        ----------
        x : torch.Tensor
            The input feature matrix.
        edge_index : torch.Tensor
            The graph connectivity matrix.
        Returns
        -------
        torch.Tensor
            The output embedding matrix.
        """
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
    """
    A GraphSAGE model used for graph embedding.
    Inherits from the torch.nn.Module class.
    Parameters
    ----------
    feat_dim : int
        The dimensionality of the input features.
    num_layers : int
        The number of hidden layers in the GraphSAGE.
    hidden_dim : int
        The dimensionality of the hidden layers.
    embed_dim : int
        The dimensionality of the output embedding.
    Attributes
    ----------
    input_layer : torch_geometric.nn.conv.SAGEConv
        The input layer of the GraphSAGE.
    hidden_layers : torch.nn.ModuleList
        A list of hidden layers in the GraphSAGE.
    output_layer : torch_geometric.nn.conv.SAGEConv or None
        The output layer of the GraphSAGE, or None if num_layers is 1.
    fc : torch.nn.Sequential or torch.nn.Linear
        The fully connected layers used to compute the embedding.
    Methods
    -------
    forward(x, edge_index)
        Computes the forward pass of the GraphSAGE.
    References
    ----------
    Hamilton, W. L., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs.
    """
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim):
        """
        Initializes a new instance of the GraphSAGE class.
        Parameters
        ----------
        feat_dim : int
            The dimensionality of the input features.
        num_layers : int
            The number of hidden layers in the GraphSAGE.
        hidden_dim : int
            The dimensionality of the hidden layers.
        embed_dim : int
            The dimensionality of the output embedding.
        """
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
        """
        Computes the forward pass of the GraphSAGE.
        Parameters
        ----------
        x : torch.Tensor
            The input feature matrix.
        edge_index : torch.Tensor
            The graph connectivity matrix.
        Returns
        -------
        torch.Tensor
            The output embedding matrix.
        """
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
    """
    A Graph Attention Network (GAT) model used for graph embedding.
    Inherits from the torch.nn.Module class.
    Parameters
    ----------
    feat_dim : int
        The dimensionality of the input features.
    num_layers : int
        The number of hidden layers in the GAT.
    hidden_dim : int
        The dimensionality of the hidden layers.
    embed_dim : int
        The dimensionality of the output embedding.
    in_heads : int
        The number of attention heads for the input layer.
    out_heads : int
        The number of attention heads for the output layer.
    Attributes
    ----------
    input_layer : torch_geometric.nn.conv.GATConv
        The input layer of the GAT.
    hidden_layers : torch.nn.ModuleList
        A list of hidden layers in the GAT.
    output_layer : torch_geometric.nn.conv.GATConv or None
        The output layer of the GAT, or None if num_layers is 1.
    fc : torch.nn.Sequential or torch.nn.Linear
        The fully connected layers used to compute the embedding.
    Methods
    -------
    forward(x, edge_index)
        Computes the forward pass of the GAT.
    References
    ----------
    Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. arXiv preprint arXiv:1710.10903.
    """
    def __init__(self, feat_dim, num_layers, hidden_dim, embed_dim, in_heads, out_heads):
        """
        Initializes a new instance of the GAT class.
        Parameters
        ----------
        feat_dim : int
            The dimensionality of the input features.
        num_layers : int
            The number of hidden layers in the GAT.
        hidden_dim : int
            The dimensionality of the hidden layers.
        embed_dim : int
            The dimensionality of the output embedding.
        in_heads : int
            The number of attention heads for the input layer.
        out_heads : int
            The number of attention heads for the output layer.
        """
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
        """
        Computes the forward pass of the GAT.
        Parameters
        ----------
        x : torch.Tensor
            The input feature matrix.
        edge_index : torch.Tensor
            The graph connectivity matrix.
        Returns
        -------
        torch.Tensor
            The output embedding matrix.
        """
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
        """
        Forwards node
        :param x: Node to use
        :return: Rectified node
        """
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
        """
        Forwards node embeddings
        :param x: Node
        :param edge_index: Edge index
        :return: Node embeddings
        """
        x = F.relu(self.input_layer(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        if self.output_layer is not None:
            x = F.relu(self.output_layer(x, edge_index))
        x = self.fc(x)

        return x