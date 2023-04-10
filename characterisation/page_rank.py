import networkx as nx
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from collections import Counter, OrderedDict
from visualisation.visualisation import *
def compute_page_rank_analysis(TN, data=False):

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

    graph = TN.get_higher_complexity()

    page_rank = nx.pagerank(graph)

    map_weighted_network(TN, custom_node_weigth=page_rank, edge_weigth=False, scale=scale, node_weight_name="Page Rank")