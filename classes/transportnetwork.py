"""
BASE GRAPH FILE

Base file for characteristics & visualisation stuff
"""

from enum import Enum
from typing import List, Tuple
import networkx as nx
from itertools import chain

__all__ = ["TransportNetwork"]

class TransportNetwork:

    graph: nx.Graph = None
    dirgraph: nx.DiGraph = None
    multigraph: nx.MultiGraph = None
    multidigraph: nx.MultiDiGraph = None

    is_spatial: bool
    pos_argument = None
    pos_dict = {}

    is_weighted: bool
    nodes_weight_argument: str

    edges_weight: str

    is_dynamic: bool
    is_interval: bool
    time_arguments: str
    time_interval_arguments: List[Tuple[str, str]]

    # def convert_multigraph_to_graph(self):
    #     """
    #     Convert a multidigraph to a multigraph
    #     """
    #     if self.multidigraph is None:
    #         raise Exception("No multidigraph to convert")
    #
    #     self.graph = self.multidigraph
    #
    #     for edge in list(self.multidigraph.edges):
    #         if edge[2] != 0:
    #             self.graph.remove_edge(edge[0], edge[1], key=edge[2])

    def __init__(self, graph,
                    dirgraph=None, \
                    multigraph=None, \
                    multidigraph=None, \
                    is_weighted=False,  \
                    nodes_weight=None, \
                    edges_weight=None, \
                    is_spatial = False, \
                    pos_argument = None, \
                    is_dynamic = False, \
                    is_interval = False):


        ### Fill the networkx grqph depending on the type of graph give by the user

        if graph.is_directed() and graph.is_multigraph():
            self.multidigraph = graph
            # self.convert_multigraph_to_graph()
        elif graph.is_directed():
            self.dirgraph = graph
        elif graph.is_multigraph():
            self.multigraph = graph
        else:
            self.graph = graph

        ## Spatial attributes ##

        if (is_spatial):
            self.is_spatial = is_spatial
            if type(pos_argument) is list and len(pos_argument) == 2:
                self.pos_argument = pos_argument
                try:
                    for node in self.multidigraph.nodes:
                        self.pos_dict[node] = (self.multidigraph.nodes[node][self.pos_argument[0]], self.multidigraph.nodes[node][self.pos_argument[1]])
                except:
                    raise NameError(f"The nodes does not have '{self.pos_argument}' attributes")
            elif type(pos_argument) is str:
                self.pos_argument = pos_argument
                self.pos_dict = nx.get_node_attributes(self.multidigraph, self.pos_argument)
                if len(self.pos_dict) == 0:
                    raise NameError(f"The nodes does not have a '{self.pos_argument}' attribute")
            else:
                raise TypeError(f"pos_argument must be a list of strings (e.g. ['lon', 'lat']) or a string (e.g. 'pos')")

        ## Weighted attributes ##
        if (is_weighted):
            self.is_weighted = is_weighted
            self.nodes_weight = nodes_weight
            self.edges_weight = edges_weight


    def get_max_lat(self):
        return max(self.pos_dict.values(), key=lambda x: x[1])[1]

    def get_min_lat(self):
        return min(self.pos_dict.values(), key=lambda x: x[1])[1]

    def get_max_lon(self):
        return max(self.pos_dict.values(), key=lambda x: x[0])[0]

    def get_min_lon(self):
        return min(self.pos_dict.values(), key=lambda x: x[0])[0]


    def __str__(self):

        graph_type = [self.graph, self.dirgraph, self.multigraph, self.multidigraph]
        string_to_print = ""

        for graph in graph_type:
            if graph is not None:
                num_nodes = self.multidigraph.number_of_nodes()
                num_edges = self.multidigraph.number_of_edges()

                string_to_print += f"Graph type: {type(self.multidigraph)}\n"
                string_to_print += f"- Number of nodes: {num_nodes}\n"
                nodes_attributes = set(chain.from_iterable(d.keys() for *_, d in self.multidigraph.nodes(data=True)))
                for attr in nodes_attributes:
                    string_to_print += " |"
                    string_to_print += f"--- {attr}\n"
                string_to_print += f"- Number of edges: {num_edges}\n"
                edges_attributes = set(chain.from_iterable(d.keys() for *_, d in self.multidigraph.edges(data=True)))
                for attr in edges_attributes:
                    string_to_print += " |"
                    string_to_print += f"--- {attr}\n"



        return string_to_print