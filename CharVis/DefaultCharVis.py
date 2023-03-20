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

def map_network(TN, spatial=True, generate_html=False, filename="map.html"):

    fig = go.Figure()

    # Define the dictionary postions depdending of is the network is spatial or not
    pos = {}
    if TN.is_spatial == False or spatial == False:
        if TN.spring_pos_dict == {}:
            TN.spring_pos_dict = nx.spring_layout(TN.graph)
        pos = TN.spring_pos_dict

    elif TN.is_spatial and spatial:
        pos = TN.pos_dict

    print(pos)

    # Define the edges and nodes positions
    edge_x = []
    edge_y = []
    for edge in TN.graph.edges():
        x0 = pos[edge[0]][0]
        y0 = pos[edge[0]][1]
        x1 = pos[edge[1]][0]
        y1 = pos[edge[1]][1]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    for node in TN.graph.nodes():
        x = pos[node][0]
        y = pos[node][1]
        node_x.append(x)
        node_y.append(y)


    if TN.is_spatial == False or spatial == False:
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines', )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                reversescale=True,
                color='antiquewhite',
                size=5,
                line_width=2))

        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=900
        )

    elif TN.is_spatial and spatial:
        edge_trace = go.Scattergeo(
            lon=edge_x, lat=edge_y,
            mode='lines',
            line=dict(width=1, color='red'),
            opacity=0.5,
        )

        node_trace = go.Scattergeo(
            lon=node_x, lat=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                reversescale=True,
                color=[],
                size=2,
                line_width=1))
        fig.update_layout(
            showlegend=False,
            geo=dict(
                projection_type='azimuthal equal area',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                lataxis=dict(range=[TN.get_min_lat() - 1, TN.get_max_lat() + 1]),
                lonaxis=dict(range=[TN.get_min_lon() - 1, TN.get_max_lon() + 1]),
            ),
            width=1200,
            height=900
        )

    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    if generate_html:
        fig.write_html(filename)

    fig.show()