from visualisation.visualisation import *
def compute_page_rank_analysis(TN, data=False):
    """
    Computes the PageRank analysis of the transport network.
    This function calculates the PageRank for each node in the network and returns
    a summary of the analysis containing the average, minimum, and maximum PageRank values.
    Optionally, it can return the raw PageRank values for each node if the 'data' parameter is set to True.
    Parameters
    ----------
    TN : TransportNetwork
        The input TransportNetwork for which the PageRank analysis will be computed.
    data : bool, optional
        If True, returns the raw PageRank values for each node. Default is False.
    Returns
    -------
    pagerank_analysis : dict
        A dictionary containing the average, minimum, and maximum PageRank values.
        If 'data' is True, returns the raw PageRank values for each node instead.
    Examples
    --------
       >>> G = ...  # Create or load a networkx graph object
       >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
       >>> pagerank_analysis = compute_page_rank_analysis(TN)
       """
    graph = TN.get_higher_complexity()

    pagerank = nx.pagerank(graph)

    if data:
        return pagerank

    avg_pagerank = sum(pagerank.values())/len(pagerank.values())
    min_pagerank = min(pagerank.values())
    max_pagerank = max(pagerank.values())

    pagerank_analysis = {
        "avg_pagerank": avg_pagerank,
        "min_pagerank": min_pagerank,
        "max_pagerank": max_pagerank
    }

    return pagerank_analysis


def map_page_rank_analysis(TN, scale=5):
    """
    Maps the PageRank analysis of the transport network.
    This function calculates the PageRank for each node in the network and creates
    a visual representation of the network with nodes scaled according to their PageRank values.
    Parameters
    ----------
    TN : TransportNetwork
        The input TransportNetwork for which the PageRank analysis will be mapped.
    scale : float, optional
        A scaling factor that determines the size of the nodes in the resulting plot.
        Default is 5.
    Returns
    -------
    None
    Examples
    --------
    >>> G = ...  # Create or load a networkx graph object
    >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
    >>> map_page_rank_analysis(TN)
    """
    graph = TN.get_higher_complexity()

    page_rank = nx.pagerank(graph)

    map_weighted_network(TN, custom_node_weigth=page_rank, edge_weigth=False, scale=scale, node_weight_name="Page Rank")