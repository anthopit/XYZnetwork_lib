import networkx as nx
def compute_assortativity_analysis(TN):
    graph = TN.get_higher_complexity()

    degree_assortativity = nx.degree_assortativity_coefficient(graph)
    assortativity_results = {
        "degree_assortativity": degree_assortativity,

     }
    if TN.is_spatial:
        spatial_assortativity = nx.attribute_assortativity_coefficient(graph, TN.pos_argument[1])

        assortativity_results["spacial_assortativity"]=spatial_assortativity

    return assortativity_results

def compute_network_indices(TN):

    graph = TN.graph

    # Compute the total number of nodes and edges
    total_nodes = graph.number_of_nodes()

    total_edges = graph.number_of_edges()

    # Calculate the beta index (Node Connectivity Index)
    beta_index = total_edges / (total_nodes - 1)

    # Calculate the gamma index (Circuit Index)
    max_edges = total_nodes * (total_nodes - 1) / 2
    gamma_index = total_edges / max_edges
    # Calculate the alpha index (Cyclic Index)
    total_cycles = nx.cycle_basis(graph)

    alpha_index = len(total_cycles) / (total_nodes - 1)

    network_indices_results = {

        "beta_index": beta_index,

        "gamma_index": gamma_index,

        "alpha_index": alpha_index,

    }

    return network_indices_results