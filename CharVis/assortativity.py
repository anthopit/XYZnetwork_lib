import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from networkviz.visualisation import *
import plotly.io as pio
from collections import Counter
import networkx as nx
from plotly.subplots import make_subplots
def compute_assortativity_analysis(TN):
    graph = TN.get_higher_complexity()

    degree_assortativity = nx.degree_assortativity_coefficient(graph)
    assortativity_results = {
        "degree_assortativity": degree_assortativity,

     }
    if TN.is_spatial:
        spatial_assortativity = nx.attribute_assortativity_coefficient(graph, TN.pos_argument[1])

        assortativity_results["spacial_assortativity"]=spatial_assortativity


    return assortativity_results

