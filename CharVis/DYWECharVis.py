import TransportNetwork as TN

class DYWEGraph(TN.GraphDefault):
    def __init__(self):
        self.graphType = TN.GraphType.DYWE