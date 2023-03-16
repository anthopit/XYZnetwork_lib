"""
BASE GRAPH FILE

Base file for characteristics & visualisation stuff
"""

from enum import Enum
import classes.transportnetwork as TN
import networkx as nx

class GraphDefault:

    graph: nx.Graph = None
    dirgraph: nx.DiGraph = None
    multigraph: nx.MultiGraph = None
    multidigraph: nx.MultiDiGraph = None

    def __init__(self):
        self.isSpatial = False
        self.isWeighted = False
        self.isDynamic = False


    def getGraphTypeStr(self):
        """
        For the moment, returns a string indicating what type of graph it is.
        FTM a placeholder, to be changed/deleted
        """
        return "Default"

    def node_edge_rel(self):
        "Output raw data"

    def plot_node_edge_rel(self):
        "Plot data given by above function"

    def cmpt_eccentricity(self):
        "Compute eccentricity, self-explanatory"

    def plot_eccentricity(self):
        "Self-explanatory"

    def show_eccentricity(self):
        "Show results of eccentricity on a map"
    def cmpt_centrality(self):
        "Compute centrality"

    def plot_centrality(self):
        "Plot centrality"

    def show_centrality(self):
        "Show centrality on map"

    def cmpt_cluster_coef(self):
        ""

    def plot_cluster_coef(self):
        ""

    def cmpt_deg_corr(self):
        ""

    def plot_deg_corr(self):
        ""

    def cmpt_assort(self):
        ""

    def cmpt_community_structure(self):
        ""

    def cmpt_deg_distribution(self):
        ""
    def plot_community_structure(self):
        ""

    def plot_deg_distribution(self):
        ""

    def plot_assort(self):
        ""


