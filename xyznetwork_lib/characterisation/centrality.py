import plotly.graph_objs as go
from plotly.subplots import make_subplots


def compute_centrality_analysis(TN, data=False):
    """
    Computes centrality analysis. Used for plotting and mapping

    :param TN: TransportNetwork
    :param data: Return data as dict?

    :return: Analysis results
    """

    graph = TN.get_higher_complexity()

    if TN.is_directed:
        in_degree_centrality = nx.in_degree_centrality(graph)
        out_degree_centrality = nx.out_degree_centrality(graph)
        eigenvector_centrality = nx.eigenvector_centrality(TN.dirgraph, max_iter=1000)
    else:
        degree_centrality = nx.degree_centrality(graph)
        eigenvector_centrality = nx.eigenvector_centrality(TN.graph, max_iter=1000)

    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)

    if data:
        centrality_resutls = {
            "eigenvector_centrality": eigenvector_centrality,
            "closeness_centrality": closeness_centrality,
            "betweenness_centrality": betweenness_centrality
        }

        if TN.is_directed:
            centrality_resutls["in_degree_centrality"] = in_degree_centrality
            centrality_resutls["out_degree_centrality"] = out_degree_centrality
        else:
            centrality_resutls["degree_centrality"] = degree_centrality

        return centrality_resutls

    if TN.is_directed:
        avg_in_degree_centrality = sum(in_degree_centrality.values())/len(in_degree_centrality.values())
        avg_out_degree_centrality = sum(out_degree_centrality.values())/len(out_degree_centrality.values())
    else:
        avg_degree_centrality = sum(degree_centrality.values())/len(degree_centrality.values())

    avg_eigenvector_centrality = sum(eigenvector_centrality.values())/len(eigenvector_centrality.values())
    avg_closeness_centrality = sum(closeness_centrality.values())/len(closeness_centrality.values())
    avg_betweenness_centrality = sum(betweenness_centrality.values())/len(betweenness_centrality.values())

    centrality_analysis = {
        "avg_eigenvector_centrality": avg_eigenvector_centrality,
        "avg_closeness_centrality": avg_closeness_centrality,
        "avg_betweenness_centrality": avg_betweenness_centrality
    }

    if TN.is_directed:
        centrality_analysis["avg_in_degree_centrality"] = avg_in_degree_centrality
        centrality_analysis["avg_out_degree_centrality"] = avg_out_degree_centrality
    else:
        centrality_analysis["avg_degree_centrality"] = avg_degree_centrality

    return centrality_analysis

def plot_centrality_analysis(TN):
    """
    Plots results from compute_centrality_analysis()
    """

    graph = TN.get_higher_complexity()

    if TN.is_directed:
        in_degree_centrality = nx.in_degree_centrality(graph)
        out_degree_centrality = nx.out_degree_centrality(graph)
        eigenvector_centrality = nx.eigenvector_centrality(TN.dirgraph, max_iter=1000)

        sorted_in_degree = sorted(in_degree_centrality.values(), reverse=True)
        sorted_out_degree = sorted(out_degree_centrality.values(), reverse=True)

        sorted_in_rank = list(range(1, len(sorted_in_degree) + 1))
        sorted_out_rank = list(range(1, len(sorted_out_degree) + 1))

    else:
        eigenvector_centrality = nx.eigenvector_centrality(TN.graph, max_iter=1000)


    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)

    sorted_degree = sorted(degree_centrality.values(), reverse=True)
    sorted_closeness = sorted(closeness_centrality.values(), reverse=True)
    sorted_betweenness = sorted(betweenness_centrality.values(), reverse=True)
    sorted_eigenvector = sorted(eigenvector_centrality.values(), reverse=True)

    sorted_degree_rank = list(range(1, len(sorted_degree) + 1))
    sorted_closeness_rank = list(range(1, len(sorted_closeness) + 1))
    sorted_betweenness_rank = list(range(1, len(sorted_betweenness) + 1))
    sorted_eigenvector_rank = list(range(1, len(sorted_eigenvector) + 1))

    # Create traces
    if TN.is_directed:
        fig = make_subplots(rows=3, cols=2, subplot_titles=("Degree Centrality", "Eigenvector Centrality", "Closeness Centrality", "Betweenness Centrality", "In-Degree Centrality", "Out-Degree Centrality"))

        fig.add_trace(go.Scatter(x=sorted_in_rank, y=sorted_in_degree,
                                    mode='lines+markers',
                                    name='in-degree'),
                        row=3, col=1)

        fig.add_trace(go.Scatter(x=sorted_out_rank, y=sorted_out_degree,
                                    mode='lines+markers',
                                    name='out-degree'),
                        row=3, col=2)

        fig.update_xaxes(title_text="rank", type='linear', row=3, col=1)
        fig.update_yaxes(title_text="in-degree", type='linear', row=3, col=1)

        fig.update_xaxes(title_text="rank", type='linear', row=3, col=2)
        fig.update_yaxes(title_text="out-degree", type='linear', row=3, col=2)

    else:
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Degree Centrality", "Eigenvector Centrality", "Closeness Centrality", "Betweenness Centrality"))


    fig.add_trace(go.Scatter(x=sorted_degree_rank, y=sorted_degree,
                                mode='lines+markers',
                                name='degree'),
                    row=1, col=1)
    fig.add_trace(go.Scatter(x=sorted_eigenvector_rank, y=sorted_eigenvector,
                                mode='lines+markers',
                                name='eigenvector'),
                    row=1, col=2)
    fig.add_trace(go.Scatter(x=sorted_closeness_rank, y=sorted_closeness,
                                mode='lines+markers',
                                name='closeness'),
                    row=2, col=1)
    fig.add_trace(go.Scatter(x=sorted_betweenness_rank, y=sorted_betweenness,
                                mode='lines+markers',
                                name='betweenness'),
                    row=2, col=2)


    fig.update_xaxes(title_text="rank", type='linear', row=1, col=1)
    fig.update_yaxes(title_text="degree", type='linear', row=1, col=1)

    fig.update_xaxes(title_text="rank", type='linear', row=1, col=2)
    fig.update_yaxes(title_text="eigenvector", type='linear', row=1, col=2)

    fig.update_xaxes(title_text="rank", type='linear', row=2, col=1)
    fig.update_yaxes(title_text="closeness", type='linear', row=2, col=1)

    fig.update_xaxes(title_text="rank", type='linear', row=2, col=2)
    fig.update_yaxes(title_text="betweenness", type='linear', row=2, col=2)

    fig.update_layout(title='Statistical distibution of centrality measures',
                      width=1200,
                      height=1000,)

    fig.show()

def map_centrality_analysis(TN, scale=5):
    """
    Maps results from compute_centrality_analysis()
    """

    graph = TN.get_higher_complexity()

    if TN.is_directed:
        eigenvector_centrality = nx.eigenvector_centrality(TN.dirgraph, max_iter=1000)
    else:

        eigenvector_centrality = nx.eigenvector_centrality(TN.graph, max_iter=1000)

    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)


    node_trace_degree, edge_trace, layout = visualisation.map_weighted_network(TN, custom_node_weigth=degree_centrality, edge_weigth=False, scale=scale, node_weight_name="degree", data=True)
    node_trace_closeness, edge_trace, layout = visualisation.map_weighted_network(TN, custom_node_weigth=closeness_centrality, edge_weigth=False, scale=scale, node_weight_name="closeness", data=True)
    node_trace_betweenness, edge_trace, layout = visualisation.map_weighted_network(TN, custom_node_weigth=betweenness_centrality, edge_weigth=False, scale=scale, node_weight_name="betweenness", data=True)
    node_trace_eigenvector, edge_trace, layout = visualisation.map_weighted_network(TN, custom_node_weigth=eigenvector_centrality, edge_weigth=False, scale=scale, node_weight_name="eigenvector", data=True)


    # Set the visible attribute of trace1 to False
    node_trace_closeness.visible = False
    node_trace_betweenness.visible = False
    node_trace_eigenvector.visible = False

    # Create the figure and add the two traces
    fig = go.Figure(data=[edge_trace, node_trace_degree, node_trace_closeness, node_trace_betweenness, node_trace_eigenvector])

    # Add a button to toggle between the two traces
    fig.update_layout(layout,
        updatemenus=[{
        'type': 'buttons',
        'showactive': True,
        'buttons': [
            {
                'label': 'degree',
                'method': 'update',
                'args': [{'visible': [True, True, False, False, False]}]
            },
            {
                'label': 'closeness',
                'method': 'update',
                'args': [{'visible': [True, False, True, False, False]}]
            },
            {
                'label': 'betweenness',
                'method': 'update',
                'args': [{'visible': [True, False, False, True, False]}]
            },
            {
                'label': 'eigenvector',
                'method': 'update',
                'args': [{'visible': [True, False, False, False, True]}]
            }

        ]
    }])

    # Show the figure
    fig.show()