def compute_page_rank_analysis(TN, data=False):
    """
    Computes page rank analysis. Used for plotting and mapping.

    :param TN: TransportNetwork
    :param data: Return data as dict?

    :return: Page rank analysis results
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
    Plots results of compute_page_rank_analysis()
    """

    graph = TN.get_higher_complexity()

    page_rank = nx.pagerank(graph)

    map_weighted_network(TN, custom_node_weigth=page_rank, edge_weigth=False, scale=scale, node_weight_name="Page Rank")