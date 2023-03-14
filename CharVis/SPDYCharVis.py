import TransportNetwork as TN

class SPDYGraph(TN.GraphDefault):
    def __init__(self):
        self.graphType = TN.GraphType.SPDY