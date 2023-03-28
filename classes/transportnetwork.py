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

    spring_pos_dict = {}

    is_spatial: bool
    pos_argument = None
    pos_dict = {}

    is_weighted: bool
    nodes_weight_argument: str
    edges_weight_argument: str

    is_distance: bool
    distance_argument: str

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
                    nodes_weight_argument=None, \
                    edges_weight_argument=None, \
                    pos_argument = None, \
                    time_arguments = None, \
                    distance_argument = None):

        if nodes_weight_argument is not None:
            if type(nodes_weight_argument) is not str:
                raise TypeError("nodes_weight_argument must be a string")
        elif edges_weight_argument is not None:
            if type(edges_weight_argument) is not str:
                raise TypeError("edges_weight_argument must be a string")
        elif pos_argument is not None:
            if type(pos_argument) is not list[str]:
                raise TypeError("pos_argument must be a list of strings")
        elif time_arguments is not None:
            if type(time_arguments) is not list[str] or type(time_arguments) is not str:
                raise TypeError("pos_argument must be a list of strings (e.g. ['lon', 'lat']) or a string (e.g. 'pos')")
        elif distance_argument is not None:
            if type(distance_argument) is not str:
                raise TypeError("distance_argument must be a string")


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

        self.is_spatial = False

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

            self.is_spatial = True

        ## Weighted attributes ##

        if (nodes_weight_argument is not None or edges_weight_argument is not None):
            self.nodes_weight_argument = nodes_weight_argument
            self.edges_weight_argument = edges_weight_argument

            self.is_weighted = True
        else:
            self.is_weighted = False
            self.nodes_weight_argument = None
            self.edges_weight_argument = None

        ## Dynamic attributes ##

        self.is_interval = False

        if (time_arguments is not None):
            if (type(time_arguments) is str):
                self.time_arguments = time_arguments
                self.is_dynamic = True
            elif (type(time_arguments) is list and len(time_arguments) == 2):
                self.time_arguments = time_arguments
                self.is_interval = True
                self.is_dynamic = True

       ## Distance attributes ##
        if (distance_argument is not None):
            self.distance_argument = distance_argument
            self.is_distance = True


    def get_max_lat(self):
        return max(self.pos_dict.values(), key=lambda x: x[1])[1]

    def get_min_lat(self):
        return min(self.pos_dict.values(), key=lambda x: x[1])[1]

    def get_max_lon(self):
        return max(self.pos_dict.values(), key=lambda x: x[0])[0]

    def get_min_lon(self):
        return min(self.pos_dict.values(), key=lambda x: x[0])[0]

    def get_max_time(self):
        if (self.is_interval):
            return max(self.multidigraph.edges(data=True), key=lambda x: x[2][self.time_arguments[1]])[2][self.time_arguments[1]]
        else:
            return max(self.multidigraph.edges(data=True), key=lambda x: x[2][self.time_arguments])[2][self.time_arguments]

    def get_min_time(self):
        if (self.is_interval):
            return min(self.multidigraph.edges(data=True), key=lambda x: x[2][self.time_arguments[0]])[2][self.time_arguments[0]]
        else:
            return min(self.multidigraph.edges(data=True), key=lambda x: x[2][self.time_arguments])[2][self.time_arguments]


    def get_node_weight_dict(self):
        if (self.nodes_weight_argument is not None):
            return nx.get_node_attributes(self.graph, self.nodes_weight_argument)
        else:
            raise Exception("No nodes weight argument associated to the graph")

    def get_edge_weight_dict(self):
        if (self.edges_weight_argument is not None):
            dict =  nx.get_edge_attributes(self.graph, self.edges_weight_argument)
            dict = {k: float(v) for k, v in dict.items()}
            return dict
        else:
            raise Exception("No edges weight argument associated to the graph")



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