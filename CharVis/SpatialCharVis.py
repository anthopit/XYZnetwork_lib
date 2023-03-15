from typing import List, Tuple
import defaultGraph as TN

class SpatialGraph(TN.GraphDefault):

    pos: List[Tuple[str, str]] = None
    pos_dict: dict = None
    def __init__(self):
        self.isSpatial = True

    def getGraphTypeStr(self):
        return "Spatial"
