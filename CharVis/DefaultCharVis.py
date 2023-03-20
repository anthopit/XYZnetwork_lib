"""
BASE GRAPH FILE

Base file for characteristics & visualisation stuff
"""

from enum import Enum
import classes.transportnetwork as TN
import networkx as nx
import plotly.graph_objects as go

class GraphDefault:

    graph: nx.Graph = None
    dirgraph: nx.DiGraph = None
    multigraph: nx.MultiGraph = None
    multidigraph: nx.MultiDiGraph = None

    def __init__(self):
        self.isSpatial = False
        self.isWeighted = False
        self.isDynamic = False


    def getGraphTypeStr(self):
        """
        For the moment, returns a string indicating what type of graph it is.
        FTM a placeholder, to be changed/deleted
        """
        return "Default"

    def node_edge_rel(self):
        "Output raw data"

    def plot_node_edge_rel(self):
        "Plot data given by above function"

    def cmpt_eccentricity(self):
        "Compute eccentricity, self-explanatory"

    def plot_eccentricity(self):
        "Self-explanatory"

    def show_eccentricity(self):
        "Show results of eccentricity on a map"
    def cmpt_centrality(self):
        "Compute centrality"

    def plot_centrality(self):
        "Plot centrality"

    def show_centrality(self):
        "Show centrality on map"

    def cmpt_cluster_coef(self):
        ""

    def plot_cluster_coef(self):
        ""

    def cmpt_deg_corr(self):
        ""

    def plot_deg_corr(self):
        ""

    def cmpt_assort(self):
        ""

    def cmpt_community_structure(self):
        ""

    def cmpt_deg_distribution(self):
        ""
    def plot_community_structure(self):
        ""

    def plot_deg_distribution(self):
        ""

    def plot_assort(self):
        ""


######################################## Default functions without OOP ########################################

def map_network(TN, spatial=True):

    print(TN)

    if TN.is_spatial and spatial:
    # if self.is_spatial == False or spatial == False:

        if TN.spring_pos_dict == {}:
            TN.spring_pos_dict = nx.spring_layout(TN.graph)

        edge_x = []
        edge_y = []
        for edge in TN.graph.edges():
            x0 = TN.spring_pos_dict[edge[0]][0]
            y0 = TN.spring_pos_dict[edge[0]][1]
            x1 = TN.spring_pos_dict[edge[1]][0]
            y1 = TN.spring_pos_dict[edge[1]][1]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines', )

        node_x = []
        node_y = []
        for node in TN.graph.nodes():
            x = TN.spring_pos_dict[node][0]
            y = TN.spring_pos_dict[node][1]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                reversescale=True,
                color='antiquewhite',
                size=5,
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                        )

        fig.update_layout(width=1200, height=900)

        fig.show()
    #
    # if self.is_spatial and spatial:

