import numpy as np
import networkx as nx
from visualisation.visualisation import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def compute_clustering_analysis(TN, data=False):
    """
    Compute the clustering analysis. Used in plot_clustering_analysis and map_clustering_analysis

    :param TN: TransportNetwork to analyse
    :param data: Return data only?

    :return: Analysis data
    :rtype: dict
    """

    if not data:
        triangles = nx.triangles(TN.graph)
        avg_triangles = sum(triangles.values())/len(triangles.values())
        transitivity = nx.transitivity(TN.graph)
        average_clustering = nx.average_clustering(TN.graph)

        clustering_analysis = {
            "avg_triangles": avg_triangles,
            "transitivity": transitivity,
            "average_clustering_coef": average_clustering
        }

        return clustering_analysis

    else:
        triangles = nx.triangles(TN.graph)

        clustering_analysis = {
            'triangles': triangles,
            'clustering': {}
        }

        for node in TN.graph.nodes():
            clustering = nx.clustering(TN.graph, nodes=node)
            clustering_analysis['clustering'][node] = clustering

        return clustering_analysis


def plot_clustering_analysis(TN):
    """
    Plot clustering analysis

    :param TN: TransportNetwork to analyse

    :return: Plotly plot of the clustering analysis data
    """
    clustering_analysis = compute_clustering_analysis(TN, data=True)

    # Get the list for the two metrics
    list_triangles = list(clustering_analysis["triangles"].values())
    list_clustering = list(clustering_analysis["clustering"].values())

    triangles_prob, _ = np.histogram(list_triangles, bins=np.arange(max(list_triangles) + 2), density=True)
    clustering_prob, _ = np.histogram(list_clustering, bins=np.arange(max(list_clustering) + 2), density=True)

    triangles_prob = triangles_prob[triangles_prob != 0]
    clustering_prob = clustering_prob[clustering_prob != 0]

    # Create traces
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Number of Triangles", "Clustering coefficient"))

    fig.add_trace(go.Scatter(x=np.arange(len(triangles_prob)), y=triangles_prob,
                             mode='lines+markers',
                             name='Triangles',
                             marker=dict(color='#e8463a')),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=np.arange(len(clustering_prob)), y=clustering_prob,
                             name='Clustering coefficient',
                             marker=dict(color='#3a80e8')),
                  row=1, col=2)

    fig.update_xaxes(title_text="Triangles", type='linear', row=1, col=1)
    fig.update_yaxes(title_text="Relative Frequency", type='linear', row=1, col=1)

    fig.update_xaxes(title_text="Clustering coefficient", type='linear', row=1, col=2)
    fig.update_yaxes(title_text="Relative Frequency", type='linear', row=1, col=2)

    fig.update_layout(title='Clustering coefficient and triangles Distribution')

    fig.show()

def map_clustering_analysis(TN, scale=5):
    """
    Maps the clustering analysis.

    :param TN: TransportNetwork
    :param scale: Scale of the mapped analysis

    :return: Plotly figure of mapped clustering analysis
    """
    clustering_analysis = compute_clustering_analysis(TN, data=True)

    print(clustering_analysis['clustering'])

    node_trace_triangles, edge_trace, layout = map_weighted_network(TN, custom_node_weigth=clustering_analysis['triangles'],
                                                                 edge_weigth=False, scale=scale,
                                                                 node_weight_name="triangles per nodes", data=True)
    node_trace_clustering_degree, edge_trace, layout = map_weighted_network(TN, custom_node_weigth=clustering_analysis['clustering'],
                                                                    edge_weigth=False, scale=scale,
                                                                    node_weight_name="clusterinmg degree per nodes", data=True)

    # Set the visible attribute of trace1 to False
    node_trace_clustering_degree.visible = False

    # Create the figure and add the two traces
    fig = go.Figure(
        data=[edge_trace, node_trace_triangles, node_trace_clustering_degree])

    # Add a button to toggle between the two traces
    fig.update_layout(layout,
                      updatemenus=[{
                          'type': 'buttons',
                          'showactive': True,
                          'buttons': [
                              {
                                  'label': 'triangles',
                                  'method': 'update',
                                  'args': [{'visible': [True, True, False]}]
                              },
                              {
                                  'label': 'clustering degree',
                                  'method': 'update',
                                  'args': [{'visible': [True, False, True]}]
                              }
                          ]
                      }])

    # Show the figure
    fig.show()