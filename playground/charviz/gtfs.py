from classes import transportnetwork as tn
import numpy as np
import networkx as nx
from preprocessing import Preprocessing as pp
from visualisation.visualisation import *

import networkx as nx
import preprocessing.Preprocessing as pp
import classes.transportnetwork as tn
from robustness_analysis.robustness import *
import pandas as pd

import multiprocessing as mp
from functools import partial


def compute_metrics(node, graph):
    # Create subgraph without current node
    subgraph = graph.copy()
    subgraph.remove_node(node)

    global_efficiency = nx.global_efficiency(subgraph)
    local_efficiency = nx.local_efficiency(subgraph)

    # Compute the average reachability
    temp_reachability = []
    for n in subgraph.nodes():
        shortest_path_length = nx.single_source_shortest_path_length(subgraph, n)
        number_of_reachable_nodes = len(shortest_path_length)

        average_node_reachability = number_of_reachable_nodes / len(subgraph.nodes())
        temp_reachability.append(average_node_reachability)

    average_reachability = sum(temp_reachability) / len(subgraph.nodes())

    # Get the list of connected components in the graph
    components = nx.connected_components(subgraph)
    # Get the size of the largest connected component
    largest_component = max(components, key=len)
    largest_size = len(largest_component)
    # Get the total number of nodes in the graph
    total_nodes = len(subgraph.nodes())
    # Compute the relative size of the largest connected component
    relative_size = largest_size / total_nodes
    giant_component_ratio = relative_size

    return node, [global_efficiency, local_efficiency, average_reachability, giant_component_ratio]


def main():
    # Load the data
    G = pp.create_network_from_trailway("../../data/Railway Data_JL.xlsx")
    TN = tn.TransportNetwork(G, pos_argument=['lon', 'lat'], time_arguments=['dep_time', 'arr_time'])
    # Assuming TN.multidigraph is a networkx multidigraph object
    graph = TN.graph

    # Print the number of nodes and edges
    print(f'Number of nodes: {graph.number_of_nodes()}')
    print(f'Number of edges: {graph.number_of_edges()}')

    # Use all available CPU cores
    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:
        compute_metrics_partial = partial(compute_metrics, graph=graph)
        results_list = list(tqdm(pool.imap_unordered(compute_metrics_partial, graph.nodes()), total=len(graph.nodes())))

    results = {node: metrics for node, metrics in results_list}

    # Save results as csv
    df = pd.DataFrame.from_dict(results, orient='index', columns=['global_efficiency', 'local_efficiency', 'average_reachability', 'giant_component_ratio'])
    df.to_csv('robustness.csv')


if __name__ == '__main__':
    main()