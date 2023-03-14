"""
BASE GRAPH FILE

Base file for characteristics & visualisation stuff
"""

from enum import Enum

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

class GraphDefault:
    graphType = GraphType.SPATIAL

    def __init__(self):
        self.graphType = GraphType.SPATIAL

    def getGraphType(self):
        return self.graphType
