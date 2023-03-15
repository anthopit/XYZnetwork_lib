"""
BASE GRAPH FILE

Base file for characteristics & visualisation stuff
"""

from enum import Enum
from typing import List, Tuple
import networkx as nx

__all__ = ["TransportNetwork"]

class GraphType(Enum):
    """
    This type will define all types of graphs.
    Each graph has a type, each case being represented below.
    For types A, B and C, we have:
    A, B, C, A&B, A&C, B&C, A&B&C
    """
    SPATIAL = 0
    DYNAMIC = 1
    WEIGHTED = 2
    SPDY = 3
    SPWE = 4
    DYWE = 5
    SDW = 6

class TransportNetwork:

    graph: nx.Graph = None
    dirgraph: nx.DiGraph = None
    multigraph: nx.MultiGraph = None
    multidigraph: nx.MultiDiGraph = None

    is_spatial: bool = False
    pos: List[Tuple[str, str]] = None
    pos_dict: dict = None

    is_weighted: bool = False
    nodes_weight: str = None
    edges_weight: str = None

    is_dynamic: bool = False
    is_interval: bool = False
    time_arguments: str = None
    time_interval_arguments: List[Tuple[str, str]] = None

    def __init__(self, graph, is_spatial=False):
        self.graph = graph
        self.is_spatial = is_spatial

