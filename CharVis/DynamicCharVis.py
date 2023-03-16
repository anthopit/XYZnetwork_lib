from typing import List, Tuple

import DefaultCharVis as TN

class DynamicGraph(TN.GraphDefault):

    is_interval: bool = False
    time_arguments: str = None
    time_interval_arguments: List[Tuple[str, str]] = None
    def __init__(self):
        self.isDynamic = True

    def getGraphTypeStr(self):
        return "Dynamic"