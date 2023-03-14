import TransportNetwork as TN

class SpatialGraph(TN.GraphDefault):
    def __init__(self):
        self.graphType = TN.GraphType.SPATIAL
