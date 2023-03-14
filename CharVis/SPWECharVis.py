import TransportNetwork as TN

class SPWEGraph(TN.GraphDefault):
    def __init__(self):
        self.graphType = TN.GraphType.SPWE