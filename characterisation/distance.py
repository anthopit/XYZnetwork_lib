import networkx as nx
from geopy.distance import distance
from visualisation.visualisation import *

def compute_distances_analysis(TN, data=False):

    if not TN.is_distance:
        raise Exception("The network does not have a distance attribute")

    if data:
        graph = TN.get_higher_complexity()
    else:
        graph = TN.graph

    # Compute the distance total sum and the average distance

    cost = 0

    euclidian_distances = {}
    real_distances = {}
    detour = {}
    for edge in list(graph.edges):
        # Compute de cost
        cost += graph.edges[edge][TN.distance_argument]


        # Compute the euclidian distance
        lat1, lon1 = graph.nodes[edge[0]][TN.pos_argument[1]], graph.nodes[edge[0]][TN.pos_argument[0]]
        lat2, lon2 = graph.nodes[edge[1]][TN.pos_argument[1]], graph.nodes[edge[1]][TN.pos_argument[0]]

        euclidian_distances[edge] = distance((lat1, lon1), (lat2, lon2)).km

        # Compute the real distance
        real_distances[edge] = graph.edges[edge][TN.distance_argument]

        if real_distances[edge] == 0:
            pass
        else:
            # Compute the detour for each edge
            detour_temp = euclidian_distances[edge] / real_distances[edge]
            if detour_temp > 1:
                if detour_temp < 1.1:
                    detour[edge] = euclidian_distances[edge] / real_distances[edge]
            else:
                detour[edge] = euclidian_distances[edge] / real_distances[edge]

    if data:
        return euclidian_distances, real_distances, detour

    # Compute the average distance
    average_distance = cost / len(graph.edges())

    average_detour = sum(detour.values()) / len(detour)

    # compute the network density
    min_lat = TN.get_min_lat()
    max_lat = TN.get_max_lat()
    min_lon = TN.get_min_lon()
    max_lon = TN.get_max_lon()

    length = distance((min_lat, min_lon), (max_lat, min_lon)).km
    width = distance((max_lat, min_lon), (max_lat, max_lon)).km

    area = length * width

    density = cost / area

    # Compute de the Pi index



    for edge in graph.edges():
        if graph.edges[edge][TN.distance_argument] < 0:
            graph.edges[edge][TN.distance_argument] = 0

    diameter = nx.diameter(graph, weight=TN.distance_argument)

    pi_index = cost / diameter

    distances_analysis = {
        "cost": cost,
        "average_distance": average_distance,
        "average_detour": average_detour,
        "density": density,
        "pi_index": pi_index
    }

    return distances_analysis


def map_detour_analysis(TN):

    if not TN.is_distance:
        raise Exception("The network does not have a distance attribute")

    euclidian_distances = {}
    real_distances = {}
    detour = {}
    for edge in TN.graph.edges():

        # Compute the euclidian distance
        lat1, lon1 = TN.graph.nodes[edge[0]][TN.pos_argument[1]], TN.graph.nodes[edge[0]][TN.pos_argument[0]]
        lat2, lon2 = TN.graph.nodes[edge[1]][TN.pos_argument[1]], TN.graph.nodes[edge[1]][TN.pos_argument[0]]

        euclidian_distances[edge] = distance((lat1, lon1), (lat2, lon2)).km
        real_distances[edge] = TN.graph.edges[edge][TN.distance_argument]

        if real_distances[edge] == 0:
            detour[edge] = 0
        else:
            # Compute the detour for each edge
            detour_temp = euclidian_distances[edge] / real_distances[edge]
            if detour_temp > 1:
                if detour_temp < 1.1:
                    detour[edge] = euclidian_distances[edge] / real_distances[edge]
                else:
                    detour[edge] = 0
            else:
                detour[edge] = euclidian_distances[edge] / real_distances[edge]


    map_weighted_network(TN, node_weigth=False, custom_edge_weigth=detour)