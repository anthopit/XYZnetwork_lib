"""
BASE GRAPH FILE

Base file for characteristics & visualisation stuff
"""

from enum import Enum
import classes.transportnetwork as tn
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from characterisation.WeightedCharVis import *
from preprocessing import Preprocessing as pp
import time

def compute_degrees(TN):
    if TN.dirgraph:
        return list(dict(TN.dirgraph.out_degree).values()), list(dict(TN.dirgraph.in_degree).values())
    else:
        return list(dict(TN.graph.degree).values())

def plot_hist_degrees(TN):
    if TN.dirgraph:
        degreesOut, degreesIn = compute_degrees(TN)
        df = pd.DataFrame({"out": degreesOut, "in": degreesIn})
        px.histogram(df).show()
    else:
        px.histogram(compute_degrees(TN)).show()

def map_degrees(TN, is_out_edges = False):
    if TN.dirgraph:
        degreesOut, degreesIn = compute_degrees(TN)
        df = pd.DataFrame({"out": degreesOut, "in": degreesIn})
        if is_out_edges:
            map_weighted_network(TN, spatial=True, scale=17, custom_node_weigth=dict(TN.dirgraph.out_degree))
        else:
            map_weighted_network(TN, spatial=True, scale=17, custom_node_weigth=dict(TN.dirgraph.in_degree))
    else:
        degrees = compute_degrees(TN)
        map_weighted_network(TN, spatial=True, scale=17, custom_node_weigth=dict(degrees))
def node_edge_rel():
    "Output raw data"

def plot_node_edge_rel():
    "Plot data given by above function"

def cmpt_eccentricity():
    "Compute eccentricity, -explanatory"

def plot_eccentricity():
    "-explanatory"

def show_eccentricity():
    "Show results of eccentricity on a map"
def cmpt_centrality():
    "Compute centrality"

def plot_centrality():
    "Plot centrality"

def show_centrality():
    "Show centrality on map"

def cmpt_cluster_coef():
    ""

def plot_cluster_coef():
    ""

def cmpt_deg_corr():
    ""

def plot_deg_corr():
    ""

def cmpt_assort():
    ""

def cmpt_community_structure():
    ""

def cmpt_deg_distribution():
    ""
def plot_community_structure():
    ""

def plot_deg_distribution():
    ""

def plot_assort():
    ""


######################################## Default functions without OOP ########################################

def map_network(TN, spatial=True, generate_html=False, filename="map.html"):

    fig = go.Figure()

    # Define the dictionary postions depdending of is the network is spatial or not
    pos = {}
    if TN.is_spatial == False or spatial == False:
        if TN.spring_pos_dict == {}:
            print("Generating spring layout, this may take a while...")
            TN.spring_pos_dict = nx.spring_layout(TN.graph)
        pos = TN.spring_pos_dict

    elif TN.is_spatial and spatial:
        pos = TN.pos_dict

    #TODO : add ineractuive etiquette

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