import defaultGraph as TN

class WeightedGraph(TN.GraphDefault):

    nodes_weight: str = None
    edges_weight: str = None
    def __init__(self):
        self.isWeighted = True

    def getGraphTypeStr(self):
        return "Weighted"