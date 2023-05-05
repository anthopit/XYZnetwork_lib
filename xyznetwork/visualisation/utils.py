import distinctipy as distipy
import matplotlib.pyplot as plt

def create_comm_colors(nb_colors):
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