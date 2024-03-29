from typing import List
import networkx as nx
from itertools import chain
from geopy.distance import distance
import warnings

__all__ = ["TransportNetwork"]

class TransportNetwork:
    """
    A class representing a transport network.
    This class is designed to handle different types of graphs (directed, undirected, multi and multi-directed) and
    store their properties, including spatial and temporal attributes, edge and node weights, and distances.
    Attributes
    ----------
    graph : nx.Graph
        The undirected graph.
    dirgraph : nx.DiGraph
        The directed graph.
    multigraph : nx.MultiGraph
        The multi-graph.
    multidigraph : nx.MultiDiGraph
        The multi-directed graph.
    is_directed : bool
        Whether the graph is directed or not.
    is_multi : bool
        Whether the graph is a multi-graph or not.
    spring_pos_dict : dict
        A dictionary containing the spring layout positions for the graph.
    is_spatial : bool
        Whether the graph has spatial attributes or not.
    pos_argument : list or str
        The spatial attributes for the graph.
    pos_dict : dict
        A dictionary containing the positions of the nodes in the graph.
    is_weighted : list
        A list containing booleans indicating if the graph has node and edge weights.
    nodes_weight_argument : str
        The attribute name for node weights.
    edges_weight_argument : str
        The attribute name for edge weights.
    nodes_weight_attr : dict
        A dictionary containing node weights.
    edges_weight_attr : dict
        A dictionary containing edge weights.
    is_distance : bool
        Whether the graph has distance attributes or not.
    distance_argument : str
        The attribute name for the distance.
    is_dynamic : bool
        Whether the graph is time-dependent or not.
    is_interval : bool
        Whether the graph has time intervals or not.
    time_arguments : str or list
        The time attributes for the graph.
    Methods
    -------
    convert_multidigraph_to_digraph()
        Convert a multi-directed graph to a directed graph.
    convert_multidirgraph_to_multigraph()
        Convert a multi-directed graph to a multi-graph.
    convert_dirgraph_to_graph()
        Convert a directed graph to an undirected graph.
    updata_graph(graph)
        Update the TransportNetwork's graph.
    __init__(graph, nodes_weight_argument, edges_weight_argument, pos_argument, time_arguments, distance_argument)
        Initialize the TransportNetwork class.
    get_higher_complexity()
        Get the graph with the highest complexity.
    get_max_lat()
        Get the maximum latitude value.
    get_min_lat()
        Get the minimum latitude value.
    get_max_lon()
        Get the maximum longitude value.
    get_min_lon()
        Get the minimum longitude value.
    get_max_time()
        Get the maximum time value.
    get_min_time()
        Get the minimum time value.
    get_node_weight_dict()
        Get the dictionary of node weights.
    get_edge_weight_dict()
        Get the dictionary of edge weights.
    __str__()
        Return a string representation of the TransportNetwork.
    """

    graph: nx.Graph = None
    dirgraph: nx.DiGraph = None
    multigraph: nx.MultiGraph = None
    multidigraph: nx.MultiDiGraph = None

    is_directed: bool = False
    is_multi: bool = False

    spring_pos_dict = {}

    is_spatial: bool
    pos_argument = None
    pos_dict = {}

    is_weighted: [bool, bool]
    nodes_weight_argument: str
    edges_weight_argument: str

    nodes_weight_attr: {}
    edges_weight_attr: {}

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

    def updata_graph(self, graph):
        """
        Updates TN's graph
        :param graph: Graph to update
        """
        if graph.is_directed() and graph.is_multigraph():
            self.multidigraph = graph
            self.multigraph = self.convert_multidirgraph_to_multigraph()
            self.dirgraph = self.convert_multidigraph_to_digraph()
            self.graph = self.convert_dirgraph_to_graph()

            self.is_directed = True
            self.is_multi = True

        elif graph.is_directed():
            self.dirgraph = graph

            self.is_directed = True

        elif graph.is_multigraph():
            self.multigraph = graph

            self.is_multi = True
        else:
            self.graph = graph


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
            if type(pos_argument) is not list or len(pos_argument) != 2:
                raise TypeError("pos_argument must be a list of strings")
        elif time_arguments is not None:
            if type(time_arguments) is not list or type(time_arguments) is not str:
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

            self.is_directed = True
            self.is_multi = True

        elif graph.is_directed():
            self.dirgraph = graph

            self.is_directed = True

        elif graph.is_multigraph():
            self.multigraph = graph

            self.is_multi = True
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

            # Add the euclidian distance for each edges as an attribute
            graph_temp = self.get_higher_complexity()
            for edge in list(graph_temp .edges):

                # Compute the euclidian distance
                lat1, lon1 = graph.nodes[edge[0]][self.pos_argument[1]], graph.nodes[edge[0]][self.pos_argument[0]]
                lat2, lon2 = graph.nodes[edge[1]][self.pos_argument[1]], graph.nodes[edge[1]][self.pos_argument[0]]

                euclidian_distances = distance((lat1, lon1), (lat2, lon2)).km

                # Add the euclidian distance as an attribute
                graph_temp.edges[edge]['euclidian_distance'] = euclidian_distances

            self.updata_graph(graph_temp)

            self.is_spatial = True

        ## Weighted attributes ##

        self.is_weighted = [False, False]

        if (nodes_weight_argument is not None) and (edges_weight_argument is not None):
            try:
                edges_weight_attr = nx.get_edge_attributes(self.graph, edges_weight_argument)
                nodes_weight_attr = nx.get_node_attributes(self.graph, nodes_weight_argument)

                if not edges_weight_attr:
                    warnings.warn(f"Edges do not have a '{edges_weight_argument}' attribute")
                else:
                    self.edges_weight_argument = edges_weight_argument
                    self.is_weighted[1] = True

                if not nodes_weight_attr:
                    warnings.warn(f"Nodes do not have a '{nodes_weight_argument}' attribute")
                else:
                    self.nodes_weight_argument = nodes_weight_argument
                    self.is_weighted[0] = True

            except AttributeError:
                pass

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

    def get_higher_complexity(self):
        if self.is_directed:
            if self.is_multi:
                return self.multidigraph
            else:
                return self.dirgraph

        else:
            if self.is_multi:
                return self.multigraph
            else:
                return self.graph

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