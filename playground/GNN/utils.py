import networkx as nx
from torch_geometric.utils import degree, one_hot
import torch.nn.functional as F
import torch
import numpy as np
import multiprocessing as mp
from networkx import NetworkXNoPath
from tqdm import tqdm

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

def cat_node_feature(data, TN, cat=True, feature='degree_one_hot', num_workers=1):

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
    elif feature == "position":
        print(num_workers)
        anchor_sets = generate_anchor_sets_networkx(TN.get_higher_complexity(), c=0.5)
        shortest_paths = compute_shortest_paths_parallel(TN.get_higher_complexity(), anchor_sets, num_workers)
        shortest_paths_list = [shortest_paths[node] for node in TN.graph.nodes]
        f = torch.tensor(shortest_paths_list, dtype=torch.float)
    # elif feature == "distance":
    #     if TN.is_distance:
    #
    #     elif TN.is_spatial:
    #     else:
    #         raise ValueError
    else:
        raise NotImplementedError

    if data.x is not None and cat:
        data.x = torch.cat([data.x, f], dim=-1)
    else:
        data.x = f

    return data

# def augment_data(data, augment_list=["edge_perturbation", "node_dropping"],node_mask_rate=0.1, edge_perturb_rate=0.1):
#
#     if "edge_perturbation" in augment_list:
#         edge_indices = torch.tensor(list(data.edge_index.T.cpu().numpy()))
#         edge_mask = torch.rand(len(edge_indices)) < edge_perturb_rate
#         edge_indices[edge_mask] = torch.randint(0, data.num_nodes, (edge_mask.sum(), 2)).to(edge_indices.device)
#         data.edge_index = edge_indices.T
#
#     if "node_dropping" in augment_list:
#         node_mask = torch.rand(data.num_nodes) < node_mask_rate
#         data.x[node_mask] = 0
#
#     return data


def augment_data(data, augment_list=["edge_perturbation", "node_dropping"], mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, edge_ratio=0.1):

    if "edge_perturbation" in augment_list:
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * edge_ratio)

        edge_index = data.edge_index.detach().clone()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(2, -1).long()


        idx_add = torch.randint(node_num, (2, perturb_num))

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        data.edge_index = new_edge_index

    if "node_dropping" in augment_list:
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()

        mask = torch.zeros(node_num)
        mask_num = int(node_num * mask_ratio)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        x[idx_mask] = torch.tensor(np.random.normal(loc=mask_mean, scale=mask_std,
                                                    size=(mask_num, feat_dim)), dtype=torch.float32)

        data.x = x

    return data

def get_random_anchorset_networkx(graph, c=0.5):
    nodes = list(graph.nodes)
    n = graph.number_of_nodes()
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_id = []

    copy_num = copy - 1
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))

        for j in range(copy - copy_num):
            anchorset_id.append(np.random.choice(nodes, size=anchor_size, replace=False))

        copy_num = copy_num - 1

    return anchorset_id

def generate_anchor_sets_networkx(graph, c=0.5):
    # Generate a list of anchor sets
    anchorset_list = get_random_anchorset_networkx(graph, c)

    return anchorset_list

def shortest_paths_worker(args):
    G, node, anchor_sets, output_dict = args
    shortest_paths = []
    for anchor_set in anchor_sets:
        anchor_set_shortest_paths = []
        for anchor in anchor_set:
            try:
                path_length = nx.shortest_path_length(G, source=node, target=anchor)
            except NetworkXNoPath:
                path_length = 100
            anchor_set_shortest_paths.append(path_length)
        shortest_paths.append(min(anchor_set_shortest_paths))
    output_dict[node] = shortest_paths

def compute_shortest_paths_parallel(G, anchor_sets, num_workers):
    nodes = list(G.nodes)
    manager = mp.Manager()
    output_dict = manager.dict()

    with mp.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(shortest_paths_worker, [(G, node, anchor_sets, output_dict) for node in nodes]), total=len(nodes)):
            pass

    return output_dict

