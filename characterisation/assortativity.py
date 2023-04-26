import networkx as nx
def compute_assortativity_analysis(TN):
    """
    Computes assortativity analysis

    :param TN: TransportNetwork

    :return: Assortativity results
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

