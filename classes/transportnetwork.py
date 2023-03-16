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
    time_arguments: str or List[str]

    def convert_multidigraph_to_digraph(self):
        """
        Convert a multidigraph to a multigraph
        """
        if self.multidigraph is None:
            raise Exception("No multidigraph to convert")
        else:
            DG = nx.DiGraph(self.multidigraph)

            return DG

    def convert_multidirgraph_to_multigraph(self):
        """
        Convert a multidigraph to a multigraph
        """
        if self.multidigraph is None:
            raise Exception("No multidigraph to convert")
        else:
            DG = nx.MultiGraph(self.multidigraph)
            DG.to_undirected()

            return DG



    def convert_dirgraph_to_graph(self):
        """
        Convert a directed graph to an undirected graph
        """
        if self.dirgraph is None:
            raise Exception("No directed graph to convert")
        else:
            G = nx.Graph(self.dirgraph)
            G.to_undirected()

            return G





    def __init__(self, graph,
                    nodes_weight=None, \
                    edges_weight=None, \
                    pos_argument = None, \
                    time_arguments = None):


        ### Fill the networkx grqph depending on the type of graph give by the user

        if graph.is_directed() and graph.is_multigraph():
            self.multidigraph = graph
            self.multigraph = self.convert_multidirgraph_to_multigraph()
            self.dirgraph = self.convert_multidigraph_to_digraph()
            self.graph = self.convert_dirgraph_to_graph()
        elif graph.is_directed():
            self.dirgraph = graph
        elif graph.is_multigraph():
            self.multigraph = graph
        else:
            self.graph = graph

        ## Spatial attributes ##

        if (pos_argument is not None):
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

            self.is_spatial = True

        ## Weighted attributes ##

        if (nodes_weight is not None or edges_weight is not None):
            self.nodes_weight = nodes_weight
            self.edges_weight = edges_weight

            self.is_weighted = True

        ## Dynamic attributes ##

        if (time_arguments is not None):
            if (type(time_arguments) is str):
                self.time_arguments = time_arguments
                self.is_interval = False
            elif (type(time_arguments) is list and len(time_arguments) == 2):
                self.time_arguments = time_arguments
                self.is_interval = True
            else:
                raise TypeError(f"time_arguments must be a list of strings (e.g. ['start', 'end']) or a string (e.g. 'time')")



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
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()

                string_to_print += f"Graph type: {type(graph)}\n"
                string_to_print += f"- Number of nodes: {num_nodes}\n"
                nodes_attributes = set(chain.from_iterable(d.keys() for *_, d in graph.nodes(data=True)))
                for attr in nodes_attributes:
                    string_to_print += " |"
                    string_to_print += f"--- {attr}\n"
                string_to_print += f"- Number of edges: {num_edges}\n"
                edges_attributes = set(chain.from_iterable(d.keys() for *_, d in graph.edges(data=True)))
                for attr in edges_attributes:
                    string_to_print += " |"
                    string_to_print += f"--- {attr}\n"



        return string_to_print