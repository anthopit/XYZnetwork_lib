import networkx as nx
from collections import Counter, OrderedDict
import plotly.graph_objs as go
import plotly.io as pio
from visualisation.visualisation import *

def compute_shortest_path_analysis(TN):
    """
     Computes the shortest path analysis of the transport network.
     This function calculates the minimum shortest path, diameter, and average shortest path length
     of the transport network.
     Parameters
     ----------
     TN : TransportNetwork
         The input TransportNetwork for which the shortest path analysis will be computed.
     Returns
     -------
     shortest_path_analysis : dict
         A dictionary containing the following keys and their corresponding values:
         - 'min_shortest_path': The minimum shortest path between any two nodes in the network.
         - 'diameter': The maximum shortest path between any two nodes in the network.
         - 'average_shortest_path_length': The average shortest path length for all pairs of nodes.
     Examples
     --------
     >>> G = ...  # Create or load a networkx graph object
     >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
     >>> shortest_path_analysis = compute_shortest_path_analysis(TN)
     """
    diameter = nx.diameter(TN.graph)
    average_shortest_path_length = nx.average_shortest_path_length(TN.graph)

    # Assuming you have already created your 'TN.graph' object
    # Compute shortest path lengths
    shortest_path_length = dict(nx.shortest_path_length(TN.graph))

    # Initialize minimum and maximum shortest path variables
    min_shortest_path = float('inf')
    max_shortest_path = 0

    # Iterate through the shortest path lengths and find the minimum and maximum values
    for source, target_lengths in shortest_path_length.items():
        for target, length in target_lengths.items():
            if source == target:  # Skip self-loops
                continue
            if length < min_shortest_path:
                min_shortest_path = length
            if length > max_shortest_path:
                max_shortest_path = length


    # Add metrics to a dictionary
    shortest_path_analysis = {
        'min_shortest_path': min_shortest_path,
        'diameter': diameter,
        'average_shortest_path_length': average_shortest_path_length,
    }

    return shortest_path_analysis

def plot_shortest_path_analysis(TN):
    """
    Plots the shortest path length distribution of the transport network.
    This function creates a scatter plot of the shortest path length distribution in the transport network.
    It also highlights the maximum and average shortest path lengths on the plot.
    Parameters
    ----------
    TN : TransportNetwork
        The input TransportNetwork for which the shortest path analysis will be plotted.
    Returns
    -------
    None
    Examples
    --------
    >>> G = ...  # Create or load a networkx graph object
    >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
    >>> plot_shortest_path_analysis(TN)
    """
    shortest_path_length = dict(nx.shortest_path_length(TN.graph))

    # Collect all shortest path lengths, excluding self-loops
    path_lengths = []
    for source, target_lengths in shortest_path_length.items():
        for target, length in target_lengths.items():
            if source != target:
                path_lengths.append(length)

    # Count the occurrences of each path length
    path_length_counts = Counter(path_lengths)

    # Prepare data for plotting
    lengths = list(path_length_counts.keys())
    counts = list(path_length_counts.values())

    # Calculate the maximum and average shortest path
    max_shortest_path = max(path_lengths)
    avg_shortest_path = sum(path_lengths) / len(path_lengths)

    # Create scatter plot traces
    main_trace = go.Scatter(x=lengths, y=counts, mode='markers+lines', marker=dict(size=8), name='Shortest Path Lengths')
    max_trace = go.Scatter(x=[max_shortest_path], y=[path_length_counts[max_shortest_path]], mode='markers', marker=dict(size=10, color='red'), name='Max Shortest Path')
    avg_trace = go.Scatter(x=[avg_shortest_path], y=[path_length_counts[int(round(avg_shortest_path))]], mode='markers', marker=dict(size=10, color='green'), name='Average Shortest Path')

    # Set the layout for the plot
    layout = go.Layout(
        title='Shortest Path Length Distribution',
        xaxis=dict(title='Shortest Path Length'),
        yaxis=dict(title='Frequency'),
    )

    # Create a Figure object
    fig = go.Figure(data=[main_trace, max_trace, avg_trace], layout=layout)

    # Show the plot
    pio.show(fig)


def map_shortest_path_analysis(TN, source_node=None, target_node=None):
    """
    Maps compute_shortest_path_analysis() results
    :param TN: TransportNetwork
    :param source_node: Source node to use
    :param target_node: Target node
    """
    # Calculate the diameter
    diameter = nx.diameter(TN.graph)

    # Find the nodes corresponding to the diameter (longest shortest path)
    longest_shortest_path = None
    max_path_length = 0
    for source, target_lengths in nx.shortest_path_length(TN.graph):
        for target, length in target_lengths.items():
            if length == diameter:
                longest_shortest_path = nx.shortest_path(TN.graph, source, target)
                max_path_length = length
                break


    edge_trace, node_trace, layout, pos = map_network(TN, data=True)

    edge_trace.showlegend = False
    node_trace.showlegend = False

    # Create the figure and add the two traces
    fig = go.Figure(data=[edge_trace, node_trace])

    if source_node is not None and target_node is not None:
        shorter_path = nx.shortest_path(TN.graph, source_node, target_node)

        # Create the positions for the edges of the shortest path
        shortest_edge_x = []
        shortest_edge_y = []
        for node in shorter_path:
            x, y = pos[node]
            shortest_edge_x.append(x)
            shortest_edge_y.append(y)

        if TN.is_spatial:
            shortest_path_trace = go.Scattergeo(
                lon=shortest_edge_x, lat=shortest_edge_y,
                line=dict(width=2, color='green'),
                hoverinfo='text',
                mode='lines',
                name='Shortest Path',
                text=f'Shortest Path between node {source_node} and {target_node}: {len(shorter_path)})'
            )
        else:
            shortest_path_trace = go.Scatter(
                x=shortest_edge_x, y=shortest_edge_y,
                line=dict(width=2, color='green'),
                hoverinfo='text',
                mode='lines',
                name='Shortest Path',
                text=f'Shortest Path between node {source} and {target}: {len(shorter_path)})'
            )

        fig.add_trace(shortest_path_trace)


    # Create the positions for the edges of the longest shortest path
    diam_edge_x = []
    diam_edge_y = []
    for node in longest_shortest_path:
        x, y = pos[node]
        diam_edge_x.append(x)
        diam_edge_y.append(y)

    if TN.is_spatial:
        diameter_path_trace = go.Scattergeo(
            lon=diam_edge_x, lat=diam_edge_y,
            line=dict(width=2, color='red'),
            hoverinfo='text',
            mode='lines',
            name='Diameter',
            text = f'Diameter: {max_path_length})'
    )

    else:
        diameter_path_trace = go.Scatter(
            x=diam_edge_x, y=diam_edge_y,
            line=dict(width=2, color='red'),
            hoverinfo='text',
            mode='lines',
            name='Diameter',
            text = f'Diameter: {max_path_length})'
        )

    fig.add_trace(diameter_path_trace)

    # Add a button to toggle between the two traces
    fig.update_layout(layout)

    # Show the figure
    fig.show()


def compute_eccentricity_analysis(TN, data=False):
    """
    Computes eccentricity analysis. Used for plotting and mapping.
    :param TN: TransportNetwork
    :return: Eccentricity analysis results
    """

    eccentricity = nx.eccentricity(TN.graph)

    if data:
        return eccentricity

    max_eccentricity = max(eccentricity.values())
    min_eccentricity = min(eccentricity.values())
    avg_eccentricity = sum(eccentricity.values()) / len(eccentricity)

    eccentricity_analysis = {
        'max_eccentricity': max_eccentricity,
        'min_eccentricity': min_eccentricity,
        'avg_eccentricity': avg_eccentricity
    }

    return eccentricity_analysis

def plot_eccentricity_analysis(TN):
    """
    Plots compute_eccentricity_analysis() results
    :param TN: TransportNetwork
    """
    # Compute the eccentricity of each node in the graph
    eccentricity = nx.eccentricity(TN.graph)

    # Count the occurrences of each eccentricity value
    eccentricity_counts = Counter(eccentricity.values())

    # Reorder the dictionary by key
    eccentricity_counts = OrderedDict(sorted(eccentricity_counts.items()))

    # Prepare data for plotting
    lengths = list(eccentricity_counts.keys())
    counts = list(eccentricity_counts.values())
    avg_eccentricity = sum(eccentricity.values()) / len(eccentricity.values())

    # Create scatter plot traces
    main_trace = go.Scatter(x=lengths, y=counts, mode='markers+lines', marker=dict(size=8),
                            name='Eccentricity')
    avg_trace = go.Scatter(x=[avg_eccentricity], y=[eccentricity_counts[int(round(avg_eccentricity))]], mode='markers',
                           marker=dict(size=10, color='green'), name='Average Eccentricity')

    # Set the layout for the plot
    layout = go.Layout(
        title='Eccentricity Distribution',
        xaxis=dict(title='Eccentricity'),
        yaxis=dict(title='Frequency'),
    )

    # Create a Figure object
    fig = go.Figure(data=[main_trace, avg_trace], layout=layout)

    # Show the plot
    pio.show(fig)

def map_eccentricity_analysis(TN, scale=5):
    """
    Maps compute_eccentricity_analysis() results
    :param TN: TransportNetwork
    :param scale: Scale of the map
    """
    # Compute the eccentricity of each node in the graph
    eccentricity = nx.eccentricity(TN.graph)

    map_weighted_network(TN, custom_node_weigth=eccentricity, edge_weigth=False, scale=scale,
                         node_weight_name="Eccentricity")