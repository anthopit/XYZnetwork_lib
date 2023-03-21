import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

# class DynamicGraph(TN.GraphDefault):
#
#     is_interval: bool = False
#     time_arguments: str = None
#     time_interval_arguments: List[Tuple[str, str]] = None
#     def __init__(self):
#         self.isDynamic = True
#
#     def getGraphTypeStr(self):
#         return "Dynamic"

def convert_minutes_to_ddhhmm(minutes):
    days = minutes // (24 * 60)
    hours = (minutes // 60) % 24
    minutes = minutes % 60
    return '{:02d}:{:02d}:{:02d}'.format(int(days), int(hours), int(minutes))

def get_gradient_color(value):
    """
    Returns a color from a gradient based on a given value.

    Parameters:
    value (float): The input value to use for determining the color.
    cmap_name (str): The name of the Matplotlib colormap to use.

    Returns:
    tuple: A tuple representing the RGB values of the color at the given value on the gradient.
    """
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(value))
    return tuple(int(x * 255) for x in rgba[:3])


def map_dynamic_network(TN, spatial=True, generate_html=False, filename="map.html", scale=1, node_weigth=True, edge_weigth=False, custom_node_weigth=None, custom_edge_weigth=None, step=None):

    if TN.is_dynamic == False:
        raise Exception("The graph is not dynamic")

    fig = go.Figure()

    if step is None:
        raise Exception("The step is not defined")

    start = TN.get_min_time()
    end = TN.get_max_time()
    step = (end - start) / step

    if custom_node_weigth is None:
        node_weigth_dict = TN.get_node_weight_dict()
        list_node_weigth = list(node_weigth_dict.values())
        list_node_weigth_scaled = [x * scale for x in list_node_weigth]
    else:
        list_node_weigth = list(custom_node_weigth.values())
        list_node_weigth_scaled = [x * scale for x in list_node_weigth]

    if custom_edge_weigth is None:
        edge_weigth_dict = TN.get_edge_weight_dict()
        list_edge_weigth = list(edge_weigth_dict.values())
        list_edge_weigth_scaled = [x * scale for x in list_edge_weigth]
    else:
        list_edge_weigth = list(custom_edge_weigth.values())
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

    # Add traces, one for each slider step
    for step in np.arange(start, end, step):
        G2 = nx.DiGraph(((source, target, attr) for source, target, attr in TN.multidigraph.edges(data=True) if
                         attr['dep_time'] < step and attr['arr_time'] > step))

        #TODO add edge weight and node weight
        
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
                line=dict(width=1, color='red'),
                opacity=0.5,
            )
        else:
            edge_trace = go.Scattergeo(
                lon=lon, lat=lat,
                mode='lines',
                line=dict(width=1, color='red'),
                opacity=0.5,
            )


        fig.add_trace(edge_trace)


    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Date time " + str(convert_minutes_to_ddhhmm(i * (end - start) / 100))}],
            # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    # node_x = []
    # node_y = []
    # for node in TN.graph.nodes():
    #     x = pos[node][0]
    #     y = pos[node][1]
    #     node_x.append(x)
    #     node_y.append(y)
    #
    # if node_weigth == True:
    #     maker_style_dict = dict(
    #             size=list_node_weigth_scaled ,
    #             color=list_node_weigth,
    #             sizemode='area',
    #             cmin=min(list_node_weigth ),
    #             cmax=max(list_node_weigth ),
    #             colorbar_title=TN.nodes_weight_argument,
    #             reversescale=True,
    #             line_width=2)
    # else:
    #     maker_style_dict = dict(
    #             sizemode='area',
    #             reversescale=True,
    #             color='#888',
    #             size=5,
    # )
    #
    # if TN.is_spatial == False or spatial == False:
    #     node_trace = go.Scatter(
    #             x=node_x, y=node_y,
    #             mode='markers',
    #             hoverinfo='text',
    #             marker=maker_style_dict
    #     )
    # else:
    #     node_trace = go.Scattergeo(
    #             lon=node_x, lat=node_y,
    #             mode='markers',
    #             hoverinfo='text',
    #             marker=maker_style_dict
    #     )
    #
    # fig.add_trace(node_trace)


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

    if generate_html:
        fig.write_html(filename)
    fig.show()


    # #TODO : add ineractuive etiquette
    #
    # # Define the edges and nodes positions
    # edge_x = []
    # edge_y = []
    # for edge in TN.graph.edges():
    #     x0 = pos[edge[0]][0]
    #     y0 = pos[edge[0]][1]
    #     x1 = pos[edge[1]][0]
    #     y1 = pos[edge[1]][1]
    #     edge_x.extend([x0, x1, None])
    #     edge_y.extend([y0, y1, None])
    #
    # node_x = []
    # node_y = []
    # for node in TN.graph.nodes():
    #     x = pos[node][0]
    #     y = pos[node][1]
    #     node_x.append(x)
    #     node_y.append(y)
    #
    # if node_weigth == True:
    #     maker_style_dict = dict(
    #             size=list_node_weigth_scaled ,
    #             color=list_node_weigth,
    #             sizemode='area',
    #             cmin=min(list_node_weigth ),
    #             cmax=max(list_node_weigth ),
    #             colorbar_title=TN.nodes_weight_argument,
    #             reversescale=True,
    #             line_width=2)
    # else:
    #     maker_style_dict = dict(
    #             sizemode='area',
    #             reversescale=True,
    #             color='#888',
    #             size=5,
    #     )
    #
    #
    #
    # if TN.is_spatial == False or spatial == False:
    #     if edge_weigth:
    #         for i, edge in enumerate(TN.graph.edges()):
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=[pos[edge[0]][0], pos[edge[1]][0]],
    #                     y=[pos[edge[0]][1], pos[edge[1]][1]],
    #                     mode='lines',
    #                     line=dict(width=1, color='rgb' + str(get_gradient_color(list_edge_weigth_scaled[i] / max(list_edge_weigth_scaled)))),
    #                     opacity=1,
    #                 )
    #             )
    #     else:
    #         edge_trace = go.Scatter(
    #             x=edge_x, y=edge_y,
    #             mode='lines',
    #             line=dict(width=1, color='#888'),
    #             opacity=0.5,
    #         )
    #
    #     node_trace = go.Scatter(
    #         x=node_x, y=node_y,
    #         mode='markers',
    #         hoverinfo='text',
    #         marker=maker_style_dict
    #     )
    #
    #     fig.update_layout(
    #         showlegend=False,
    #         hovermode='closest',
    #         margin=dict(b=20, l=5, r=5, t=40),
    #         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #         width=1200,
    #         height=900
    #     )
    #
    # elif TN.is_spatial and spatial:
    #     if edge_weigth:
    #         for i, edge in enumerate(TN.graph.edges()):
    #             fig.add_trace(
    #                 go.Scattergeo(
    #                     lon=[pos[edge[0]][0], pos[edge[1]][0]],
    #                     lat=[pos[edge[0]][1], pos[edge[1]][1]],
    #                     mode='lines',
    #                     line=dict(width=1, color='rgb' + str(get_gradient_color(list_edge_weigth_scaled[i] / max(list_edge_weigth_scaled)))),
    #                     opacity=1,
    #                 )
    #             )
    #     else:
    #         edge_trace = go.Scattergeo(
    #             lon=edge_x, lat=edge_y,
    #             mode='lines',
    #             line=dict(width=1, color='#888'),
    #             opacity=0.5,
    #         )
    #
    #     node_trace = go.Scattergeo(
    #         lon=node_x, lat=node_y,
    #         mode='markers',
    #         hoverinfo='text',
    #         marker=maker_style_dict
    #     )
    #     fig.update_layout(
    #         showlegend=False,
    #         geo=dict(
    #             projection_type='azimuthal equal area',
    #             showland=True,
    #             landcolor='rgb(243, 243, 243)',
    #             countrycolor='rgb(204, 204, 204)',
    #             lataxis=dict(range=[TN.get_min_lat() - 1, TN.get_max_lat() + 1]),
    #             lonaxis=dict(range=[TN.get_min_lon() - 1, TN.get_max_lon() + 1]),
    #         ),
    #         width=1200,
    #         height=900
    #     )
    #
    # if not edge_weigth:
    #     fig.add_trace(edge_trace)
    # fig.add_trace(node_trace)
    #
    # if generate_html:
    #     fig.write_html(filename)
    #
    # fig.show()