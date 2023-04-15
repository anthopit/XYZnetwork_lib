import networkx as nx
from torch_geometric.utils import degree, one_hot
import torch.nn.functional as F
import torch
def get_max_deg(data):
    """
    Find the max degree across all nodes in graphs.
    """
    max_deg = 0

    row, col = data.edge_index
    num_nodes = data.num_nodes
    deg = degree(row, num_nodes)
    deg = max(deg).item()
    if deg > max_deg:
        max_deg = int(deg)
    return max_deg

def cat_node_feature(data, TN, cat=True, feature='degree_one_hot'):

    if feature == 'degree_one_hot':
        # Compute the degree of each node
        # and one-hot encode it
        max_degree = get_max_deg(data)
        idx = data.edge_index[0]
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
        f = deg
    elif feature == 'one_hot':
        indices = torch.tensor([i for i in range(data.num_nodes)])
        one_hot_encoding = one_hot(indices, num_classes=data.num_nodes)
        f = one_hot_encoding
    elif feature == "constant":
        f = torch.ones(data.num_nodes, 1)
    elif feature == "pagerank":
        G = TN.get_higher_complexity()
        pagerank = nx.pagerank(G)
        pagerank_features = [pagerank[node] for node in G.nodes()]
        f = torch.tensor(pagerank_features).view(-1, 1).float()
    elif feature == "degree":
        G = TN.get_higher_complexity()
        degree_c = nx.degree_centrality(G)
        degree_features = [degree_c[node] for node in G.nodes()]
        f = torch.tensor(degree_features).view(-1, 1).float()
    elif feature == "betweenness":
        G = TN.get_higher_complexity()
        centrality = nx.betweenness_centrality(G)
        centrality_features = [centrality[node] for node in G.nodes()]
        f = torch.tensor(centrality_features).view(-1, 1).float()
    elif feature == "closeness":
        G = TN.get_higher_complexity()
        centrality = nx.closeness_centrality(G)
        centrality_features = [centrality[node] for node in G.nodes()]
        f = torch.tensor(centrality_features).view(-1, 1).float()
    elif feature == "eigenvector":
        G = TN.graph
        centrality = nx.eigenvector_centrality(G)
        centrality_features = [centrality[node] for node in G.nodes()]
        f = torch.tensor(centrality_features).view(-1, 1).float()
    elif feature == "clustering":
        G = TN.graph
        clustering_coefficient = nx.clustering(G)
        clustering_coefficient_features = [clustering_coefficient[node] for node in G.nodes()]
        f = torch.tensor(clustering_coefficient_features).view(-1, 1).float()
    else:
        raise NotImplementedError

    if data.x is not None and cat:
        data.x = torch.cat([data.x, f], dim=-1)
    else:
        data.x = f

    return data

def augment_data(data, augment_list=["edge_perturbation", "node_dropping"],node_mask_rate=0.2, edge_perturb_rate=0.2):

    if "edge_perturbation" in augment_list:
        edge_indices = torch.tensor(list(data.edge_index.T.cpu().numpy()))
        edge_mask = torch.rand(len(edge_indices)) < edge_perturb_rate
        edge_indices[edge_mask] = torch.randint(0, data.num_nodes, (edge_mask.sum(), 2)).to(edge_indices.device)
        data.edge_index = edge_indices.T

    if "node_dropping" in augment_list:
        node_mask = torch.rand(data.num_nodes) < node_mask_rate
        data.x[node_mask] = 0


    return data