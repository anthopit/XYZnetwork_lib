from geopy.distance import distance


def compute_distances_analysis(TN, data=False, unit="km"):
    """
     Computes various distance-related metrics for the transport network.

     This function calculates the total cost, average distance, average detour,
     density, and Pi index for the transport network.

     Parameters
     ----------
     TN : TransportNetwork
         The input TransportNetwork for which the distance analysis will be computed.
     data : bool, optional
         If True, returns the data for euclidian distances, real distances, and detours,
         otherwise returns the computed metrics, by default False.
     unit : str, optional
         Unit for distance measurement, "km" for kilometers or "m" for meters, by default "km".

     Returns
     -------
     dict or tuple
         A dictionary containing distance analysis metrics, or a tuple containing
         euclidian distances, real distances, and detours if data is True.

     Raises
     ------
     Exception
         If the network does not have a distance attribute or the unit is not valid.

     Examples
     --------
     >>> G = ...  # Create or load a networkx graph object
     >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
     >>> compute_distances_analysis(TN, data=False, unit="km")
     """
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

        if unit == "km":
            euclidian_distances[edge] = distance((lat1, lon1), (lat2, lon2)).km
        elif unit == "m":
            euclidian_distances[edge] = distance((lat1, lon1), (lat2, lon2)).m
        else:
            raise Exception("The unit is not valid")

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

    if unit == "km":
        length = distance((min_lat, min_lon), (max_lat, min_lon)).km
        width = distance((max_lat, min_lon), (max_lat, max_lon)).km
    elif unit == "m":
        length = distance((min_lat, min_lon), (max_lat, min_lon)).m
        width = distance((max_lat, min_lon), (max_lat, max_lon)).m

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
    """
    Visualizes the detour analysis of the transport network.

    This function computes the detour for each edge in the network and plots a
    map of the network where the edges are colored based on their detour values.
    The detour is calculated as the ratio of the Euclidean distance to the real
    distance between the nodes of each edge.

    Parameters
    ----------
    TN : TransportNetwork
        The input TransportNetwork for which the detour analysis will be visualized.

    Raises
    ------
    Exception
        If the network does not have a distance attribute.

    Examples
    --------
    >>> G = ...  # Create or load a networkx graph object
    >>> TN = tn.TransportNetwork(G)  # Create a TransportNetwork object
    >>> map_detour_analysis(TN)
    """
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