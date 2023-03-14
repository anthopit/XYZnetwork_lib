import TransportNetwork as TN

class DynamicGraph(TN.GraphDefault):
    def __init__(self):
        self.graphType = TN.GraphType.DYNAMIC