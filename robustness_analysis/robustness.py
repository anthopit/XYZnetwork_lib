from math import *
import random
import networkx as nx
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from robustness_analysis.utils import *


def compute_robustness_analysis(TN, node_attack=True, edge_attack=False, precision=0.1, attack_type=['random', 'degree', 'closeness', 'betweenness', 'eigenvector']):
    r"""
    References
    ----------
    .. [1] Kammer, Frank and Hanjo Taubig. Graph Connectivity. in Brandes and
        Erlebach, 'Network Analysis: Methodological Foundations', Lecture
        Notes in Computer Science, Volume 3418, Springer-Verlag, 2005.
        http://www.informatik.uni-augsburg.de/thi/personen/kammer/Graph_Connectivity.pdf
    """

    mask_attack = ['random', 'degree', 'closeness', 'betweenness', 'eigenvector']
    compare_arrays(mask_attack, attack_type, 'This type of attack is not supported')

    centrality_graph = TN.get_higher_complexity()

    if TN.is_directed and edge_attack:
        graph = TN.dirgraph
    elif node_attack or edge_attack:
        graph = TN.graph
    else:
        raise Exception("Only one of the two parameters node_attack and edge_attack can be True")

    # Compute the number of nodes to remove per iteration
    number_to_remove = ceil(len(graph.nodes()) * precision)

    robustness_analysis = {}
    for attack in tqdm(attack_type, desc="Processing attack:"):
        subgraph = graph.copy()

        node_list = []
        if attack == 'random':
            # Get the list of all nodes
            node_list = list(subgraph.nodes())
        elif attack == 'degree':
            degree_centralities = nx.degree_centrality(centrality_graph)
            node_list = sorted(degree_centralities.items(), key=lambda x: x[1], reverse=True)
            node_list = [node[0] for node in node_list]
        elif attack == 'closeness':
            closeness_centralities = nx.closeness_centrality(centrality_graph)
            node_list = sorted(closeness_centralities.items(), key=lambda x: x[1], reverse=True)
            node_list = [node[0] for node in node_list]
        elif attack == 'betweenness':
            betweenness_centralities = nx.betweenness_centrality(centrality_graph)
            node_list = sorted(betweenness_centralities.items(), key=lambda x: x[1], reverse=True)
            node_list = [node[0] for node in node_list]
        elif attack == 'eigenvector':
            eigenvector_centralities = nx.eigenvector_centrality(TN.graph, max_iter=10000)
            node_list = sorted(eigenvector_centralities.items(), key=lambda x: x[1], reverse=True)
            node_list = [node[0] for node in node_list]


        # Iterate while the pourcentage of node removed is less than 100%
        total_removed = 0
        percentage_disrupted = []
        efficiency = []
        subgraph_ratio = []
        giant_component_ratio = []
        average_reachability = []

        # Compute the percentage of subgraph
        subgraph_ratio.append(nx.number_connected_components(subgraph) / len(graph.nodes()))

        # Compute the global network efficiency
        efficiency.append(round(nx.global_efficiency(subgraph), 6))

        # Get the list of connected components in the graph
        components = nx.connected_components(subgraph)
        # Get the size of the largest connected component
        largest_component = max(components, key=len)
        largest_size = len(largest_component)
        # Get the total number of nodes in the graph
        total_nodes = len(graph.nodes())
        # Compute the relative size of the largest connected component
        relative_size = largest_size / total_nodes
        giant_component_ratio.append(relative_size)

        # Compute the average reachability
        temp_reachability = []
        for node in tqdm(subgraph.nodes()):
            shortest_path_length = nx.single_source_shortest_path_length(subgraph, node)
            number_of_reachable_nodes = len(shortest_path_length)

            average_node_reachability = number_of_reachable_nodes / len(graph.nodes())
            temp_reachability.append(average_node_reachability)

        percentage_disrupted.append(0)


        average_reachability.append(sum(temp_reachability) / len(graph.nodes()))

        for i in tqdm(range(0, len(graph.nodes()) - len(graph.nodes()) % number_to_remove, number_to_remove)):
            total_removed += number_to_remove

            ################################ Random node attack ################################
            # Randomly select nodes to remove
            if attack == 'random':
                nodes_to_remove = random.sample(node_list, number_to_remove)
            else:
                nodes_to_remove = node_list[:number_to_remove]

            # Remove the selected nodes from the graph
            for node in nodes_to_remove:
                subgraph.remove_node(node)

            # Update the list of nodes
            node_list = list(subgraph.nodes())

            ####################################################################################

            # Compute the percentage of subgraph
            subgraph_ratio.append((nx.number_connected_components(subgraph) +  total_removed)/ len(graph.nodes()))

            # Compute the global network efficiency
            efficiency.append(round(nx.global_efficiency(subgraph), 6))

            # Compute the relative size of the giant component
            # Get the list of connected components in the graph
            components = nx.connected_components(subgraph)
            # Get the size of the largest connected component
            try:
                largest_component = max(components, key=len)
                largest_size = len(largest_component)
                # Get the total number of nodes in the graph
                total_nodes = len(graph.nodes())
                # Compute the relative size of the largest connected component
                relative_size = largest_size / total_nodes
                giant_component_ratio.append(relative_size)
            except:
                giant_component_ratio.append(0)

            # Compute the average reachability
            # Compute the average reachability
            temp_reachability = []
            for node in tqdm(subgraph.nodes()):
                shortest_path_length = nx.single_source_shortest_path_length(subgraph, node)
                number_of_reachable_nodes = len(shortest_path_length)

                average_node_reachability = number_of_reachable_nodes / len(graph.nodes())
                temp_reachability.append(average_node_reachability)

            average_reachability.append(sum(temp_reachability) / len(graph.nodes()))

            # Compute the porucentage of nodes removed
            percentage = total_removed / len(graph.nodes()) * 100
            percentage_disrupted.append(percentage)


        # Compute the number of nodes to remove for the last iteration
        last_number_to_remove = len(graph.nodes()) % number_to_remove

        # Randomly select nodes to remove
        nodes_to_remove = random.sample(node_list, last_number_to_remove)

        # Remove the selected nodes from the graph
        for node in nodes_to_remove:
            subgraph.remove_node(node)

        # Compute the percentage of subgraph
        subgraph_ratio.append(1)

        # Compute the global network efficiency
        efficiency.append(round(nx.global_efficiency(subgraph), 6))

        giant_component_ratio.append(0)

        # Compute the average reachability
        average_reachability.append(0)


        total_removed += last_number_to_remove
        percentage = total_removed / len(graph.nodes()) * 100
        percentage_disrupted.append(percentage)

        robustness_analysis["percentage_disrupted"] = percentage_disrupted
        robustness_analysis["efficiency_" + attack] = efficiency
        robustness_analysis["subgraph_ratio_" + attack] = subgraph_ratio
        robustness_analysis["giant_component_ratio_" + attack] = giant_component_ratio
        robustness_analysis["average_reachability_" + attack] = average_reachability


    return robustness_analysis


def plot_robustness_analysis(TN, node_attack=True, edge_attack=False, precision=0.1, attack_type=['random', 'degree', 'closeness', 'betweenness', 'eigenvector']):
    robustness_analysis = compute_robustness_analysis(TN, node_attack=node_attack, edge_attack=edge_attack, precision=precision, attack_type=attack_type)

    marker_symbol = {
        'random' : 'circle',
        'degree' : 'square',
        'closeness' : 'diamond',
        'betweenness' : 'cross',
        'eigenvector' : 'x'
    }

    marker_color = {
        'random' : '#e34646',
        'degree' : '#64e38a',
        'closeness' : '#b464e3',
        'betweenness' : '#64ace3',
        'eigenvector' : '#fcba03'
    }
    fig = make_subplots(rows=2, cols=2)

    for attack in attack_type:
        # Plot each metrics
        fig.add_trace(go.Scatter(x=robustness_analysis["percentage_disrupted"], y=robustness_analysis["subgraph_ratio_" + attack],
                             mode='markers',
                             name=attack,
                             legendgroup=attack,
                             marker=dict(
                             symbol=marker_symbol[attack],
                             color=marker_color[attack])),
                  row=1, col=1)

        fig.add_trace(go.Scatter(x=robustness_analysis["percentage_disrupted"], y=robustness_analysis["efficiency_" + attack],
                             mode='markers',
                                 name=attack,
                                 legendgroup=attack,
                                 showlegend=False,
                                 marker=dict(
                                     symbol=marker_symbol[attack],
                                     color=marker_color[attack])),
                  row=2, col=1)

        fig.add_trace(go.Scatter(x=robustness_analysis["percentage_disrupted"], y=robustness_analysis["giant_component_ratio_" + attack],
                             mode='markers',
                                 name=attack,
                                 legendgroup=attack,
                                 showlegend=False,
                                 marker=dict(
                                     symbol=marker_symbol[attack],
                                     color=marker_color[attack])),
                  row=1, col=2)

        fig.add_trace(go.Scatter(x=robustness_analysis["percentage_disrupted"], y=robustness_analysis["average_reachability_" + attack],
                             mode='markers',
                                 name=attack,
                                 legendgroup=attack,
                                 showlegend=False,
                                 marker=dict(
                                     symbol=marker_symbol[attack],
                                     color=marker_color[attack])),
                  row=2, col=2)


        fig.update_xaxes(title_text="Link disrupted (%)", type='linear', row=1, col=1)
        fig.update_yaxes(title_text="Ratio of Sub-graph", type='linear', row=1, col=1)

        fig.update_xaxes(title_text="Link disrupted (%)", type='linear', row=1, col=2)
        fig.update_yaxes(title_text="Relative size of Largest Component", type='linear', row=1, col=2)

        fig.update_xaxes(title_text="Link disrupted (%)", type='linear', row=2, col=1)
        fig.update_yaxes(title_text="Global Efficiency", type='linear', row=2, col=1)

        fig.update_xaxes(title_text="Link disrupted (%)k", type='linear', row=2, col=2)
        fig.update_yaxes(title_text="Average Reachability", type='linear', row=2, col=2)

        fig.update_layout(title='Resilience again node failure analysis',
                          width=1200,
                          height=900,
                          )

    fig.show()


def map_robustness_analysis(TN, node_attack=True, edge_attack=True, attack_type='random', precision=0.1, custome_node_list=None):
    fig = go.Figure()

    # Define the dictionary postions depdending of is the network is spatial or not
    pos = {}
    if not TN.is_spatial:
        if TN.spring_pos_dict == {}:
            print("Generating spring layout, this may take a while...")
            TN.spring_pos_dict = nx.spring_layout(TN.graph)
        pos = TN.spring_pos_dict

    elif TN.is_spatial:
        pos = TN.pos_dict

    if custome_node_list is not None:
        # create a copy of the graph with all the custom nodes removed
        graph = TN.graph.copy()
        graph.remove_nodes_from(custome_node_list)

        # Add the remover nodes to the node trace in red
        # Define the node trace
        node_x = []
        node_y = []
        for node in custome_node_list:
            x = pos[node][0]
            y = pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if TN.is_spatial == False:
            remove_node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='red',
                    size=5,
                )
            )
        else:
            remove_node_trace = go.Scattergeo(
                lon=node_x, lat=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='red',
                    size=5,
                )
            )

        for node in graph.nodes():
            x = pos[node][0]
            y = pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if TN.is_spatial == False:
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#888',
                    size=3,
                )
            )
        else:
            node_trace = go.Scattergeo(
                lon=node_x, lat=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#888',
                    size=3,
                )
            )

        lat = []
        lon = []
        for edge in graph.edges():
            x0 = pos[edge[0]][0]
            y0 = pos[edge[0]][1]
            x1 = pos[edge[1]][0]
            y1 = pos[edge[1]][1]
            lon.extend([x0, x1, None])
            lat.extend([y0, y1, None])

        if TN.is_spatial == False:
            edge_trace = go.Scatter(
                x=lon, y=lat,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )
        else:
            edge_trace = go.Scattergeo(
                lon=lon, lat=lat,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )

        # Get all the nodes wihtout any edges
        isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]


        # Add the remover nodes to the node trace in red
        # Define the node trace
        node_x = []
        node_y = []
        for node in isolated_nodes:
            x = pos[node][0]
            y = pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if TN.is_spatial == False:
            isolated_node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#4c0191',
                    size=5,
                )
            )
        else:
            isolated_node_trace = go.Scattergeo(
                lon=node_x, lat=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#4c0191',
                    size=5,
                )
            )

        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)
        fig.add_trace(remove_node_trace)
        fig.add_trace(isolated_node_trace)

        # Define layout of the map
        fig.update_layout(
            showlegend=False,
            geo=dict(
                projection_type='azimuthal equal area',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                lataxis=dict(range=[TN.get_min_lat() - 1, TN.get_max_lat() + 1]),  # set the latitude range to [40, 60]
                lonaxis=dict(range=[TN.get_min_lon() - 1, TN.get_max_lon() + 1]),
                # set the longitude range to [-10, 20]
            ),
            width=1200,
            height=900,
        )

        fig.show()

        return 0


    mask_attack = ['random', 'degree', 'closeness', 'betweenness', 'eigenvector']
    compare_arrays(mask_attack, [attack_type], 'This type of attack is not supported')

    if TN.is_directed and edge_attack:
        graph = TN.dirgraph
    elif node_attack or edge_attack:
        graph = TN.graph
    else:
        raise Exception("Only one of the two parameters node_attack and edge_attack can be True")

    centrality_graph = TN.get_higher_complexity()

    node_list = []
    if attack_type == 'random':
        # Get the list of all nodes
        node_list = list(graph.nodes())
    elif attack_type == 'degree':
        degree_centralities = nx.degree_centrality(centrality_graph)
        node_list = sorted(degree_centralities.items(), key=lambda x: x[1], reverse=True)
        node_list = [node[0] for node in node_list]
    elif attack_type == 'closeness':
        closeness_centralities = nx.closeness_centrality(centrality_graph)
        node_list = sorted(closeness_centralities.items(), key=lambda x: x[1], reverse=True)
        node_list = [node[0] for node in node_list]
    elif attack_type == 'betweenness':
        betweenness_centralities = nx.betweenness_centrality(centrality_graph)
        node_list = sorted(betweenness_centralities.items(), key=lambda x: x[1], reverse=True)
        node_list = [node[0] for node in node_list]
    elif attack_type == 'eigenvector':
        eigenvector_centralities = nx.eigenvector_centrality(TN.graph, max_iter=1000)
        node_list = sorted(eigenvector_centralities.items(), key=lambda x: x[1], reverse=True)
        node_list = [node[0] for node in node_list]

    # Define the node trace
    node_x = []
    node_y = []
    for node in TN.graph.nodes():
        x = pos[node][0]
        y = pos[node][1]
        node_x.append(x)
        node_y.append(y)

    if TN.is_spatial == False:
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                reversescale=True,
                color='#888',
                size=3,
            )
        )
    else:
        node_trace = go.Scattergeo(
            lon=node_x, lat=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                reversescale=True,
                color='#888',
                size=3,
            )
        )

    lat = []
    lon = []
    for edge in graph.edges():
        x0 = pos[edge[0]][0]
        y0 = pos[edge[0]][1]
        x1 = pos[edge[1]][0]
        y1 = pos[edge[1]][1]
        lon.extend([x0, x1, None])
        lat.extend([y0, y1, None])

    if TN.is_spatial == False:
        edge_trace = go.Scatter(
            x=lon, y=lat,
            mode='lines',
            line=dict(width=1, color='#888'),
            opacity=0.5,
        )
    else:
        edge_trace = go.Scattergeo(
            lon=lon, lat=lat,
            mode='lines',
            line=dict(width=1, color='#888'),
            opacity=0.5,
        )

    # Add node trace as the first layer of the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    # Compute the number of nodes to remove per iteration
    number_to_remove = ceil(len(graph.nodes()) * precision)

    total_removed = 0
    percentage_disrupted = []
    percentage_disrupted.append(0)
    for i in tqdm(range(0, len(graph.nodes()) - len(graph.nodes()) % number_to_remove, number_to_remove)):

        # Define the node trace
        node_x = []
        node_y = []
        for node in graph.nodes():
            x = pos[node][0]
            y = pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if TN.is_spatial == False:
            current_node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#888',
                    size=3,
                )
            )
        else:
            current_node_trace = go.Scattergeo(
                lon=node_x, lat=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#888',
                    size=3,
                )
            )


        total_removed += number_to_remove

        # Randomly select nodes to remove
        if attack_type == 'random':
            nodes_to_remove = random.sample(node_list, number_to_remove)
        else:
            nodes_to_remove = node_list[:number_to_remove]

        # Remove the selected nodes from the graph
        for node in nodes_to_remove:
            graph.remove_node(node)

        # Update the list of nodes
        node_list = list(graph.nodes())


        # Compute the porucentage of nodes removed
        percentage = total_removed / len(graph.nodes()) * 100
        percentage_disrupted.append(percentage)

        # Add the remover nodes to the node trace in red
        # Define the node trace
        node_x = []
        node_y = []
        for node in nodes_to_remove:
            x = pos[node][0]
            y = pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if TN.is_spatial == False:
            remove_node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='red',
                    size=3,
                )
            )
        else:
            remove_node_trace = go.Scattergeo(
                lon=node_x, lat=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='red',
                    size=3,
                )
            )

        lat = []
        lon = []
        for edge in graph.edges():
            x0 = pos[edge[0]][0]
            y0 = pos[edge[0]][1]
            x1 = pos[edge[1]][0]
            y1 = pos[edge[1]][1]
            lon.extend([x0, x1, None])
            lat.extend([y0, y1, None])

        if TN.is_spatial == False:
            edge_trace = go.Scatter(
                x=lon, y=lat,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )
        else:
            edge_trace = go.Scattergeo(
                lon=lon, lat=lat,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )

        # Get all the nodes wihtout any edges
        isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]

        if len(isolated_nodes) == 0:
            isolated_node_trace = go.Scatter(
                x=[], y=[],
            )
        else:
            # Add the remover nodes to the node trace in red
            # Define the node trace
            node_x = []
            node_y = []
            for node in isolated_nodes:
                x = pos[node][0]
                y = pos[node][1]
                node_x.append(x)
                node_y.append(y)

            if TN.is_spatial == False:
                isolated_node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                        sizemode='area',
                        reversescale=True,
                        color='#4c0191',
                        size=5,
                    )
                )
            else:
                isolated_node_trace = go.Scattergeo(
                    lon=node_x, lat=node_y,
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='#4c0191',
                    size=5,
                )
            )

        fig.add_trace(edge_trace)
        fig.add_trace(current_node_trace)
        fig.add_trace(remove_node_trace)
        fig.add_trace(isolated_node_trace)

    # Define the node trace
    node_x = []
    node_y = []
    for node in graph.nodes():
        x = pos[node][0]
        y = pos[node][1]
        node_x.append(x)
        node_y.append(y)

    if TN.is_spatial == False:
        current_node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                reversescale=True,
                color='#888',
                size=3,
            )
        )
    else:
        current_node_trace = go.Scattergeo(
            lon=node_x, lat=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                reversescale=True,
                color='#888',
                size=3,
            )
        )

        # Add the remover nodes to the node trace in red
        # Define the node trace
        node_x = []
        node_y = []
        for node in node_list:
            x = pos[node][0]
            y = pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if TN.is_spatial == False:
            remove_node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='red',
                    size=3,
                )
            )
        else:
            remove_node_trace = go.Scattergeo(
                lon=node_x, lat=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    sizemode='area',
                    reversescale=True,
                    color='red',
                    size=3,
                )
            )

        lat = []
        lon = []
        for edge in graph.edges():
            x0 = pos[edge[0]][0]
            y0 = pos[edge[0]][1]
            x1 = pos[edge[1]][0]
            y1 = pos[edge[1]][1]
            lon.extend([x0, x1, None])
            lat.extend([y0, y1, None])

        if TN.is_spatial == False:
            edge_trace = go.Scatter(
                x=lon, y=lat,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )
        else:
            edge_trace = go.Scattergeo(
                lon=lon, lat=lat,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )

        fig.add_trace(edge_trace)
        fig.add_trace(edge_trace)
        fig.add_trace(current_node_trace)
        fig.add_trace(remove_node_trace)



    steps = []
    step = dict(
        method="update",
        args=[{"visible": [True] + [True] + [False] * (len(fig.data) - 2)},
              ],
    )
    steps.append(step)
    for i in range(2, len(fig.data), 4):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  ],
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i + 1] = True
        step["args"][0]["visible"][i + 2] = True
        step["args"][0]["visible"][i + 3] = True

        steps.append(step)

    # Create and add slider
    sliders = [dict(
        currentvalue={"prefix": "Percentage of disrupted node: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    if TN.is_spatial == False:
        layout = dict(showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      width=1200,
                      height=900
                      )
    else:
        layout = dict(showlegend=False,
            geo=dict(
                projection_type='azimuthal equal area',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                lataxis=dict(range=[TN.get_min_lat() - 1, TN.get_max_lat() + 1]),  # set the latitude range to [40, 60]
                lonaxis=dict(range=[TN.get_min_lon() - 1, TN.get_max_lon() + 1]),
                # set the longitude range to [-10, 20]
            ),
            width=1200,
            height=900,
        )

    # Define layout of the map
    fig.update_layout(
        layout
    )

    fig.show()

