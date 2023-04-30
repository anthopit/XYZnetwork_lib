import distinctipy as distipy

def create_comm_colors(nb_colors):
    """
    Create a list of colors for the communities
    :param communities: list of communities
    :return: list of colors
    """
    colors = distipy.get_colors(nb_colors)
    colors = [tuple([i * 255 for i in c]) for c in colors]
    # convert rgb tuple to hex
    colors = [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in colors]

    return colors

def get_node_position(pos, graph, weight_dict=None):
    node_x = []
    node_y = []
    txt = []
    for node in graph.nodes():
        x = pos[node][0]
        y = pos[node][1]
        node_x.append(x)
        node_y.append(y)
        if weight_dict is not None:
            txt.append(f'{node}: {weight_dict[node]}')
        else:
            txt.append(f'Node: {node}')

    return node_x, node_y, txt

def get_edge_position(pos, graph):
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0 = pos[edge[0]][0]
        y0 = pos[edge[0]][1]
        x1 = pos[edge[1]][0]
        y1 = pos[edge[1]][1]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    return edge_x, edge_y