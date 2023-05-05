import networkx as nx
def compute_assortativity_analysis(TN):
    """
    Compute the assortativity analysis for a given network.
    This function calculates the degree assortativity coefficient and, if the network
    is spatial, the spatial assortativity coefficient. It returns a dictionary containing
    the assortativity analysis results.
    Parameters
    ----------
    TN : Network
        The input network for which the assortativity analysis will be computed.
    Returns
    -------
    assortativity_results : dict
        A dictionary containing the assortativity analysis results. The keys are:
        - "degree_assortativity": The degree assortativity coefficient of the network.
        - "spacial_assortativity": The spatial assortativity coefficient of the network
                                   (only present if the network is spatial).
    Examples
    --------
    >>> TN = tn.TransportNetwork(G) # Create or load a Network object
    >>> results = compute_assortativity_analysis(TN)
    >>> print(results)
    {'degree_assortativity': 0.12345, 'spacial_assortativity': -0.09876}
    References
    ----------
    .. [1] Newman, M. E. J. (2002). Assortative mixing in networks. Physical Review Letters, 89(20), 208701.
           https://doi.org/10.1103/PhysRevLett.89.208701
    """

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
    """
    Compute the network indices for a given network.
    This function calculates the beta index (Node Connectivity Index), gamma index (Circuit Index),
    and alpha index (Cyclic Index) for the input network. It returns a dictionary containing the
    network indices results.
    Parameters
    ----------
    TN : Network
        The input network for which the network indices will be computed.
    Returns
    -------
    network_indices_results : dict
        A dictionary containing the network indices results. The keys are:
        - "beta_index": The node connectivity index of the network.
        - "gamma_index": The circuit index of the network.
        - "alpha_index": The cyclic index of the network.
    Examples
    --------
    >>> TN = tn.TransportNetwork(G) # Create or load a Network object
    >>> results = compute_network_indices(TN)
    >>> print(results)
    {'beta_index': 1.25, 'gamma_index': 0.3, 'alpha_index': 0.6}
    """

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