import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualisation.utils import *

def convert_minutes_to_ddhhmm(minutes):
    total_minutes = int(minutes)
    days = total_minutes // (60 * 24)
    remaining_minutes = total_minutes % (60 * 24)
    hours = remaining_minutes // 60
    minutes = remaining_minutes % 60
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


def get_gradient_color(value):
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(value))
    return tuple(int(x * 255) for x in rgba[:3])

def map_network(TN, spatial=True, generate_html=False, filename="map.html", data=False):
    """
    Create a visual representation of the network.

    This function creates a visual representation of the network using the Plotly package. The network can be
    displayed using a spring layout or a spatial layout, depending on the input parameters.

    Parameters
    ----------
    TN : object
        Network object.
    spatial : bool, optional
        If True, the network will be displayed using a spatial layout; otherwise, a spring layout will be used.
        Default is True.
    generate_html : bool, optional
        If True, the function will generate an HTML file with the plot. Default is False.
    filename : str, optional
        Filename for the generated HTML file. Default is "map.html".
    data : bool, optional
        If True, the function will return the edge_trace, node_trace, layout, and pos instead of showing the plot.
        Default is False.

    Returns
    -------
    None or tuple
        None if data is False, or a tuple (edge_trace, node_trace, layout, pos) if data is True.

    """
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

    edge_x, edge_y = get_edge_position(pos, TN.graph)

    node_x, node_y, txt = get_node_position(pos, TN.graph)


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

        layout = dict(
            showlegend=True,
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
                line_width=1),
            text = txt)


        layout = dict(
            showlegend=True,
            geo=dict(
                projection_type='azimuthal equal area',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                lataxis=dict(range=[TN.get_min_lat(), TN.get_max_lat()]),
                lonaxis=dict(range=[TN.get_min_lon(), TN.get_max_lon()]),
                resolution=50,
            ),
            width=1200,
            height=900
        )

    if data:
        edge_trace.line.color = 'grey'
        return edge_trace, node_trace, layout, pos

    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    fig.update_layout(
        layout
    )

    if generate_html:
        fig.write_html(filename)

    fig.show()


def map_weighted_network(TN, spatial=True, generate_html=False, filename="map.html", scale=1, node_weigth=True, edge_weigth=True, custom_node_weigth=None, custom_edge_weigth=None, node_size=0, discrete_color=False, data=False, node_weight_name="Custom"):

    fig = go.Figure()

    if node_weigth:
        if custom_node_weigth is None:
            node_weigth_dict = TN.get_node_weight_dict()
            list_node_weigth = list(node_weigth_dict.values())
            list_node_weigth_normalized = [(x - min(list_node_weigth)) / (max(list_node_weigth) - min(list_node_weigth)) for x in list_node_weigth]
            list_node_weigth_scaled = [x * scale * 40 for x in list_node_weigth_normalized]
            weight_name = TN.nodes_weight_argument
        else:
            node_weigth_dict = custom_node_weigth
            list_node_weigth = list(node_weigth_dict.values())
            list_node_weigth_normalized = [(x - min(list_node_weigth)) / (max(list_node_weigth) - min(list_node_weigth)) for x in list_node_weigth]
            list_node_weigth_scaled = [x * scale * 40 for x in list_node_weigth_normalized]
            weight_name = node_weight_name


    if edge_weigth:
        if custom_edge_weigth is None and edge_weigth:
            edge_weigth_dict = TN.get_edge_weight_dict()
            list_edge_weigth = list(edge_weigth_dict.values())
            list_edge_weigth_scaled = [x * scale for x in list_edge_weigth]
        else:
            edge_weigth_dict = custom_edge_weigth
            list_edge_weigth = list(edge_weigth_dict.values())
            list_edge_weigth_scaled = [x * scale for x in list_edge_weigth]


    # Define the dictionary postions depdending of is the network is spatial or not
    pos = {}
    if TN.is_spatial == False or spatial == False:
        if TN.spring_pos_dict == {}:
            print("Generating spring layout, this may take a while...")
            TN.spring_pos_dict = nx.spring_layout(TN.graph)
        pos = TN.spring_pos_dict

    elif TN.is_spatial and spatial:
        pos = TN.pos_dict

    edge_x, edge_y = get_edge_position(pos, TN.graph)

    if node_weigth:
        node_x, node_y, txt = get_node_position(pos, TN.graph, node_weigth_dict)
    else:
        node_x, node_y, txt = get_node_position(pos, TN.graph)

    if discrete_color:
        comm_colors = create_comm_colors(len(set(list_node_weigth)))
        node_colors = [comm_colors[node_weigth_dict[node]-1] for node in TN.graph.nodes()]


    if node_weigth == True:
        maker_style_dict = dict(
                size= list_node_weigth_scaled if not node_size else node_size,
                color=list_node_weigth if not discrete_color else node_colors,
                sizemode='area',
                cmin=min(list_node_weigth ) if not discrete_color else None,
                cmax=max(list_node_weigth ) if not discrete_color else None,
                colorbar_title=weight_name if not discrete_color else None,
                reversescale=False)
    else:
        maker_style_dict = dict(
                sizemode='area',
                reversescale=True,
                color='#888',
                size=5,
        )



    if TN.is_spatial == False or spatial == False:
        if edge_weigth:
            for i, edge in enumerate(TN.graph.edges()):
                fig.add_trace(
                    go.Scatter(
                        x=[pos[edge[0]][0], pos[edge[1]][0]],
                        y=[pos[edge[0]][1], pos[edge[1]][1]],
                        hoverinfo='text',
                        mode='lines',
                        line=dict(width=1, color='rgb' + str(get_gradient_color(list_edge_weigth_scaled[i] / max(list_edge_weigth_scaled)))),
                        opacity=1,
                        text = f'{edge[0]} - {edge[1]} : {edge_weigth_dict[edge]}'
                    )
                )
        else:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=maker_style_dict,
            text = txt
        )

        layout = dict(showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=900
       )



    elif TN.is_spatial and spatial:
        if edge_weigth:
            for i, edge in enumerate(TN.graph.edges()):
                fig.add_trace(
                    go.Scattergeo(
                        lon=[pos[edge[0]][0], pos[edge[1]][0]],
                        lat=[pos[edge[0]][1], pos[edge[1]][1]],
                        hoverinfo='text',
                        mode='lines',
                        line=dict(width=1, color='rgb' + str(get_gradient_color(list_edge_weigth_scaled[i] / max(list_edge_weigth_scaled)))),
                        opacity=1,
                        text=f'Detour: oui'
                    )
                )
        else:
            edge_trace = go.Scattergeo(
                lon=edge_x, lat=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
            )

        node_trace = go.Scattergeo(
            lon=node_x, lat=node_y,
            mode='markers',
            hoverinfo='text',
            marker=maker_style_dict,
            text = txt
        )

        layout = dict(showlegend=False,
                geo=dict(
                projection_type='azimuthal equal area',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                lataxis=dict(range=[TN.get_min_lat(), TN.get_max_lat()]),
                lonaxis=dict(range=[TN.get_min_lon(), TN.get_max_lon()]),
                resolution=50,
            ),
            width=1200,
            height=900
        )

    if data:
        return node_trace, edge_trace, layout



    if not edge_weigth:
        fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    fig.update_layout(
        layout
    )

    if generate_html:
        fig.write_html(filename)

    fig.show()


def map_dynamic_network(TN, spatial=True, generate_html=False, filename="map.html", step=None):
    """
    Create a visual representation of the weighted network.

    This function creates a visual representation of the weighted network using the Plotly package. The network can be
    displayed using a spring layout or a spatial layout, depending on the input parameters. It supports custom node and
    edge weights, as well as discrete colors for nodes.

    Parameters
    ----------
    TN : object
        Network object.
    spatial : bool, optional
        If True, the network will be displayed using a spatial layout; otherwise, a spring layout will be used.
        Default is True.
    generate_html : bool, optional
        If True, the function will generate an HTML file with the plot. Default is False.
    filename : str, optional
        Filename for the generated HTML file. Default is "map.html".
    scale : float, optional
        Scale factor for node and edge sizes. Default is 1.
    node_weigth : bool, optional
        If True, node weights will be displayed. Default is True.
    edge_weigth : bool, optional
        If True, edge weights will be displayed. Default is True.
    custom_node_weigth : dict, optional
        A dictionary with custom node weights. Default is None.
    custom_edge_weigth : dict, optional
        A dictionary with custom edge weights. Default is None.
    node_size : int, optional
        Fixed node size. Default is 0 (variable node size).
    discrete_color : bool, optional
        If True, discrete colors will be used for nodes. Default is False.
    data : bool, optional
        If True, the function will return the node_trace, edge_trace, and layout instead of showing the plot.
        Default is False.
    node_weight_name : str, optional
        Name of the custom node weight. Default is "Custom".

    Returns
    -------
    None or tuple
        None if data is False, or a tuple (node_trace, edge_trace, layout) if data is True.

    """
    if TN.is_dynamic == False:
        raise Exception("The graph is not dynamic")

    fig = go.Figure()

    if step is None:
        raise Exception("The step is not defined")

    step_num = step
    start = TN.get_min_time()
    end = TN.get_max_time()
    step = (end - start) / step

    # Define the dictionary postions depdending of is the network is spatial or not
    pos = {}
    if TN.is_spatial == False or spatial == False:
        if TN.spring_pos_dict == {}:
            print("Generating spring layout, this may take a while...")
            TN.spring_pos_dict = nx.spring_layout(TN.graph)
        pos = TN.spring_pos_dict

    elif TN.is_spatial and spatial:
        pos = TN.pos_dict

    node_x, node_y, txt = get_node_position(pos, TN.graph)

    if TN.is_spatial == False or spatial == False:
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',
                reversescale=True,
                color='#888',
                size=5,
            ),
            text = txt
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
                size=5,
            ),
            text=txt
        )

    # Add node trace as the first layer of the figure
    fig.add_trace(node_trace)

    # Add traces, one for each slider step
    for step in np.arange(start, end+1, step):
        if step <= start:
            step = start
        if step >= end:
            step = end

        G2 = nx.DiGraph(((source, target, attr) for source, target, attr in TN.multidigraph.edges(data=True) if
                         attr[TN.time_arguments[0]] <= step and attr[TN.time_arguments[1]] >= step))

        # sub_network_edges_attr = nx.get_edge_attributes(G2, TN.edge_weight_attribute)
        # sub_network_nodes_attr_list = list(nx.get_node_attributes(G2, TN.node_weight_attribute).values())
        lat = []
        lon = []
        for edge in G2.edges():
            x0 = pos[edge[0]][0]
            y0 = pos[edge[0]][1]
            x1 = pos[edge[1]][0]
            y1 = pos[edge[1]][1]
            lon.extend([x0, x1, None])
            lat.extend([y0, y1, None])

        if TN.is_spatial == False or spatial == False:
            edge_trace = go.Scatter(
                x=lon, y=lat,
                mode='lines',
                line=dict(width=2, color='red'),
                opacity=0.5,
            )
        else:
            edge_trace = go.Scattergeo(
                lon=lon, lat=lat,
                mode='lines',
                line=dict(width=2, color='red'),
                opacity=0.5,
            )

        fig.add_trace(edge_trace)

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [True] + [False] * len(fig.data)},
                  {"title": "Date time " + str(convert_minutes_to_ddhhmm((i+1) * ((end - start) / step_num)))}],
            # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Create and add slider
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    # Define layout of the map
    fig.update_layout(
        showlegend=False,
        geo=dict(
            projection_type='azimuthal equal area',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            lataxis=dict(range=[TN.get_min_lat(), TN.get_max_lat()]),
            lonaxis=dict(range=[TN.get_min_lon(), TN.get_max_lon()]),
            resolution=50,
            # set the longitude range to [-10, 20]
        ),
        width=1200,
        height=900,
    )

    if generate_html:
        fig.write_html(filename)
    fig.show()

    # #TODO : add ineractuive etiquette




def plot_tsne_embedding(emb_df, node_cluster=None):

    if node_cluster is not None:
        comm_colors = create_comm_colors(len(set(node_cluster.values())))
        node_colors = [comm_colors[node_cluster[node] - 1] for node in node_cluster.keys()]


    tsne = TSNE(n_components=2)

    if isinstance(emb_df, dict):
        tsne_result = tsne.fit_transform(emb_df.values)
    else:
        tsne_result = tsne.fit_transform(emb_df)

    # Create a Plotly scatter plot
    scatter_plot = go.Scatter(
        x=tsne_result[:,0],  # X values
        y=tsne_result[:,1],  # Y values
        mode='markers',  # Set the mode to markers to create a scatter plot
        marker=dict(
            size=5,  # Set the size of the markers
            opacity=0.8,  # Set the opacity of the markers
            color = node_colors if node_cluster is not None else 'blue'
        )
    )

    # Set the title of the plot
    layout = go.Layout(
        title='Node2vec Embeddings Scatter Plot',
        width=1200,
        height=900,
    )

    # Create a Plotly figure with the scatter plot and the layout
    fig = go.Figure(data=[scatter_plot], layout=layout)

    # Show the Plotly figure in a browser
    fig.show()