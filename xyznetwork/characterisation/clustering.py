from visualisation.visualisation import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def compute_clustering_analysis(TN, data=False):
    """
    Computes clustering coefficients and triangles for the input transport network.
    This function calculates the average number of triangles, transitivity, and average clustering
    coefficient for the input transport network. It can also return the data for each node.
    Parameters
    ----------
    TN : TransportNetwork
        The input TransportNetwork for which the clustering analysis has been computed.
    data : bool, optional
        If True, the function returns the number of triangles and clustering coefficient for each node.
        If False (default), the function returns the average values.
    Returns
    -------
    dict
        A dictionary containing the clustering analysis results.
    Examples
    --------
    >>> G = ...  # Create or load a networkx graph object
    >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
    >>> compute_clustering_analysis(TN)
    {'avg_triangles': ..., 'transitivity': ..., 'average_clustering_coef': ...}
    >>> compute_clustering_analysis(TN, data=True)
    {'triangles': {node1: ..., node2: ..., ...}, 'clustering': {node1: ..., node2: ..., ...}}
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
    Plots the distribution of clustering coefficients and number of triangles for the input transport network.
    This function creates a plot of the distribution of clustering coefficients and the number of triangles
    for the input transport network.
    Parameters
    ----------
    TN : TransportNetwork
        The input TransportNetwork for which the clustering analysis has been computed.
    Returns
    -------
    None
    Examples
    --------
    >>> G = ...  # Create or load a networkx graph object
    >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
    >>> plot_clustering_analysis(TN)
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
     Plots the triangles and clustering degree per node on a map for the input transport network.
     This function creates an interactive map of the input transport network, displaying the triangles and
     clustering degree for each node. The node size on the map is scaled based on the metric value.
     Parameters
     ----------
     TN : TransportNetwork
         The input TransportNetwork for which the clustering analysis has been computed.
     scale : float, optional
         A scaling factor for the size of the nodes on the map, by default 5.
     Returns
     -------
     None
     Examples
     --------
     >>> G = ...  # Create or load a networkx graph object
     >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
     >>> map_clustering_analysis(TN)
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