import TransportNetwork as TN

class WeightedGraph(TN.GraphDefault):
    def __init__(self):
        self.graphType = TN.GraphType.WEIGHTED