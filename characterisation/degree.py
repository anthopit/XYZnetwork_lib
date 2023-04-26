import networkx as nx
from visualisation.visualisation import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

def compute_node_degree_analysis(TN, data=False):
    """
    Computes node degree analysis. Used for plotting & mapping

    :param TN: TransportNetwork
    :param data: Return only data of analysis?

    :return: Analysis data
    """

    if TN.is_directed:
        if TN.is_multi:
            graph = TN.multidigraph
        else:
            graph = TN.dirgraph

        degree_analysis = {}
        for node in graph.nodes():
            degree_analysis[node] = [graph.in_degree(node), graph.out_degree(node)]

        if data:
            return degree_analysis

        total_in_degree = sum(dict(graph.in_degree()).values())
        min_in_degree = min(dict(graph.in_degree()).values())
        max_in_degree = max(dict(graph.in_degree()).values())
        min_out_degree = min(dict(graph.out_degree()).values())
        max_out_degree = max(dict(graph.out_degree()).values())
        avg_degree = sum(dict(graph.out_degree()).values())/len(dict(graph.out_degree()).values())

        node_degree_analysis = {
            "min_in_degree": min_in_degree,
            "max_in_degree": max_in_degree,
            "min_out_degree": min_out_degree,
            "max_out_degree": max_out_degree,
            "avg_out_degree": avg_degree
        }

        return node_degree_analysis

    elif not TN.is_directed:
        if TN.is_multi:
            graph = TN.multigraph
        else:
            graph = TN.graph

        degree_analysis = {}
        for node in graph.nodes():
            degree_analysis[node] = [graph.degree(node)]

        if data:
            return degree_analysis

        min_degree = min(dict(graph.degree()).values())
        max_degree = max(dict(graph.degree()).values())
        avg_degree = sum(dict(graph.degree()).values())/len(dict(graph.degree()).values())

        node_degree_analysis = {
            "min_degree": min_degree,
            "max_degree": max_degree,
            "avg_degree": avg_degree
        }

        return node_degree_analysis

    else:
        raise ValueError("There is no graph to compute the degree analysis")



def plot_distribution_degree_analysis(TN):
    """
    Plots results of degree analysis

    :return: Plotted results of the analysis
    """
    if TN.is_directed:
        if TN.is_multi:
            graph = TN.multidigraph
        else:
            graph = TN.dirgraph

        in_degrees = list(dict(graph.in_degree()).values())
        out_degrees = list(dict(graph.out_degree()).values())

        in_degree_prob, _ = np.histogram(in_degrees, bins=np.arange(max(in_degrees)+2), density=True)
        out_degree_prob, _ = np.histogram(out_degrees, bins=np.arange(max(out_degrees)+2), density=True)

        in_degree_prob = in_degree_prob[in_degree_prob != 0]
        out_degree_prob = out_degree_prob[out_degree_prob != 0]

        # Create traces
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Normal Scale", "Log Scale"))

        fig.add_trace(go.Scatter(x=np.arange(len(in_degree_prob)), y=in_degree_prob,
                                 mode='markers',
                                 name='in-degree',
                                 marker=dict(color='#e8463a')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(out_degree_prob)), y=out_degree_prob,
                                 mode='markers',
                                 name='out-degree',
                                 marker = dict(color='#3a80e8')),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(len(in_degree_prob)), y=in_degree_prob,
                                 mode='markers',
                                 name='in-degree (log)',
                                 marker=dict(color='#e8463a')),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=np.arange(len(out_degree_prob)), y=out_degree_prob,
                                 mode='markers',
                                 name='out-degree (log)',
                                 marker=dict(color='#3a80e8')),
                      row=1, col=2)

        fig.update_xaxes(title_text="k", type='linear', row=1, col=1)
        fig.update_yaxes(title_text="P(k)", type='linear', row=1, col=1)

        fig.update_xaxes(title_text="k", type='log', row=1, col=2)
        fig.update_yaxes(title_text="P(k)", type='log', row=1, col=2)

        fig.update_layout(title='Degree Distribution Probability',
                          width=1400,
                          height=700,
                          )

        fig.show()

    elif not TN.is_directed:
        if TN.is_multi:
            graph = TN.multigraph
        else:
            graph = TN.graph

        degrees = list(dict(graph.degree()).values())

        degree_prob, _ = np.histogram(degrees, bins=np.arange(max(degrees)+2), density=True)

        degree_prob = degree_prob[degree_prob != 0]

        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(degree_prob)), y=degree_prob,
                                 mode='lines+markers'))

        fig.update_layout(title='Degree Distribution Probability',
                          xaxis=dict(title='P(k)'),
                          yaxis=dict(title='k'))

        fig.show()

    else:
        raise ValueError("There is no graph to compute the degree analysis")



def map_node_degree_analysis(TN, scale=5):
    """
    Maps results of degree analysis

    :return: Plotly map of the analysis results
    """

    if TN.is_directed:
        if TN.is_multi:
            graph = TN.multidigraph
        else:
            graph = TN.dirgraph


        node_trace_1, edge_trace, layout = map_weighted_network(TN, custom_node_weigth=dict(graph.in_degree()), edge_weigth=False, scale=scale, node_weight_name="Node degree", data=True)
        node_trace_2, edge_trace, layout  = map_weighted_network(TN, custom_node_weigth=dict(graph.out_degree()), edge_weigth=False, scale=scale, node_weight_name="Node degree", data=True)

        # Set the visible attribute of trace1 to False
        node_trace_2.visible = False

        # Create the figure and add the two traces
        fig = go.Figure(data=[edge_trace, node_trace_1, node_trace_2])

        # Add a button to toggle between the two traces
        fig.update_layout(layout,
            updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {
                    'label': 'in-degree',
                    'method': 'update',
                    'args': [{'visible': [True, True, False]}]
                },
                {
                    'label': 'out-degree',
                    'method': 'update',
                    'args': [{'visible': [True, False, True]}]
                }
            ]
        }])

        # Show the figure
        fig.show()

    elif not TN.is_directed:
        if TN.is_multi:
            graph = TN.multigraph
        else:
            graph = TN.graph

        map_weighted_network(TN, custom_node_weigth=dict(graph.degree()), edge_weigth=False)

    else:
        raise ValueError("There is no graph to compute the degree analysis")