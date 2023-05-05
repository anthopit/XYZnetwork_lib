import pandas as pd
import networkx as nx
from tqdm import tqdm
from preprocessing.utils import convertTimetoMinute, getTrainMaxSpeed

def create_network_from_trailway(path):
    """
    Create a network graph from a trailway data file.
    This function reads an Excel file containing trailway data, creates a network graph using the NetworkX package,
    and processes the data to include node and edge attributes.
    Parameters
    ----------
    path : str
        Path to the Excel file containing the trailway data.
    Returns
    -------
    G : nx.MultiDiGraph
        The created network graph.
    """
    df = pd.read_excel(path)
    df = df.replace(" ", 0)
    G = nx.MultiDiGraph()


    st_no_comp = 0
    prev_node = 0
    prev_mileage = 0
    prev_dep_time = 0
    print("Network creation: ")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not G.has_node(row["st_id"]):
            G.add_node(row["st_id"], lon=row["lon"], lat=row["lat"])
        if row["st_no"] == st_no_comp:
            G.add_edge(prev_node, row["st_id"], \
                       dep_time=prev_dep_time, \
                       arr_time=convertTimetoMinute(row["arr_time"], row["date"]), \
                       train=row["train"], \
                       train_max_speed=getTrainMaxSpeed(row["train"]),
                       day=row["date"], \
                       distance=row["mileage"] - prev_mileage)

            st_no_comp = row["st_no"] + 1
            prev_node = row["st_id"]
            prev_mileage = row["mileage"]
            prev_dep_time = convertTimetoMinute(row["dep_time"], row["date"])

        else:
            prev_dep_time = convertTimetoMinute(row["dep_time"], row["date"])
            st_no_comp = row["st_no"] + 1
            prev_node = row["st_id"]
            prev_mileage = 0

    # OUTLIERS PROCESSING #

    # Add distance for edges with distance < 0
    # To do that, we find an other edge wich relies on the same nodes and has a distance > 0
    # then we set the distance of the edge with distance < 0 to the distance of the edge with distance > 0
    for u in G.edges(data=True):
        if u[2]["distance"] < 0:
            for v in G.edges([u[0], u[1]], data=True):
                if v[1] == u[0] or v[1] == u[1]:
                    if v[2]["distance"] > 0:
                        u[2]["distance"] = v[2]["distance"]
                        break

    return G


def create_network_from_GTFS(path):
    """
    Create a network graph from GTFS data files.
    This function reads GTFS data files containing public transport information, creates a network graph using the
    NetworkX package, and processes the data to include node and edge attributes.
    Parameters
    ----------
    path : str
        Path to the directory containing the GTFS data files (stop_times.txt, stops.txt, trips.txt, and shapes.txt).
    Returns
    -------
    G : nx.MultiDiGraph
        The created network graph.
    """
    df_stop_time = pd.read_csv(path + '/stop_times.txt')
    df_stop = pd.read_csv(path + '/stops.txt')
    df_trip = pd.read_csv(path + '/trips.txt')
    df_shape = pd.read_csv(path + '/shapes.txt')

    # Merge the two dataframes by keeping the stop sequence order
    global_transport_df = pd.merge(df_stop_time, df_stop, on='stop_id', how='left', sort=False)
    gtfs_df = pd.merge(global_transport_df, df_trip, on='trip_id', how='left', sort=False)

    G = nx.MultiDiGraph()

    st_no_comp = 0
    prev_node = 0
    prev_mileage = 0
    prev_dep_time = 0
    print("Network creation: ")
    for index, row in tqdm(gtfs_df.iterrows(), total=gtfs_df.shape[0]):
        # Remove the last character of the stop_id
        stop_id = row["stop_id"][:-1]
        if not G.has_node(stop_id):
            G.add_node(stop_id, lon=row["stop_lon"], lat=row["stop_lat"])
        if row["stop_sequence"] == st_no_comp:
            G.add_edge(prev_node, stop_id, \
                       arrival_time=convertTimetoMinute(row["arrival_time"]),
                       departure_time=prev_dep_time, \
                       trip_id=row["trip_id"], \
                       distance=row["shape_dist_traveled"] - prev_mileage,
                       route_id=row["route_id"])

            st_no_comp = row["stop_sequence"] + 1
            prev_node = stop_id
            prev_mileage = row["shape_dist_traveled"]
            prev_dep_time = convertTimetoMinute(row["departure_time"])

        else:
            prev_dep_time = convertTimetoMinute(row["departure_time"])
            st_no_comp = row["stop_sequence"] + 1
            prev_node = stop_id
            prev_mileage = 0

    return G

def create_network_from_edges(path):
    """
    Create a network graph from an edge list file.
    This function reads an edge list file containing node connections, creates a network graph using the
    NetworkX package, and adds nodes and edges to the graph.
    Parameters
    ----------
    path : str
        Path to the edge list file.
    Returns
    -------
    G : nx.Graph
        The created network graph.
    """
    #Read data from file
    with open(path, "r") as f:
        lines = f.readlines()

    # Create a graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for line in tqdm(lines):
        if not line.startswith("%"):
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)

    return

