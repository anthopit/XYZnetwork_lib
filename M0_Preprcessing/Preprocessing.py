
import networkx as nx

def create_network_from_edges(path):

    #Read data from file
    with open(path, "r") as f:
        lines = f.readlines()

    # Create a graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for line in lines:
        if not line.startswith("%"):
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)

