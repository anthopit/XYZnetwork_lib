import pandas as pd
import networkx as nx
from tqdm import tqdm
from datetime import datetime, timedelta


############################################# Utils #############################################

def getTrainMaxSpeed(train_id):
    train_id = str(train_id)
    if train_id[0].isdigit():
        return "80"
    elif train_id[0].isalpha():
        if train_id[0] == "G" or train_id[0] == "C":
            return "350"
        elif train_id[0] == "D":
            return "260"
        elif train_id[0] == "Z" or train_id[0] == "T":
            return "160"
        else:
            return "120"


def convertTimetoMinute(time, day):

    """
    Process outliers time datas
    Some arrive and depart time are in fraction of a day, so we need to convert them to minutes
    ex: 0.8884 = 21:30:00
    """

    try:
        time_float = float(time)
        # Convert the fraction of a day to a timedelta object
        delta = timedelta(days=time_float)
        start_of_day = datetime(year=1, month=1, day=1)
        time = start_of_day + delta
    except:
        pass

    if day == "Day 1":
        minutes = time.hour * 60 + time.minute
    elif day == "Day 2":
        minutes = time.hour * 60 + time.minute + 24 * 60
    elif day == "Day 3":
        minutes = time.hour * 60 + time.minute + 24 * 60 * 2
    elif day == "Day 4":
        minutes = time.hour * 60 + time.minute + 24 * 60 * 3

    return minutes

###################################### Network ###############################################

def create_network_from_trailway(path):

    df = pd.read_excel(path)
    G = nx.MultiDiGraph()

    st_no_comp = 0
    prev_node = 0
    prev_mileage = 0
    prev_dep_time = 0
    print("Network creation: ")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not G.has_node(row["st_id"]):
            G.add_node(row["st_id"], pos=(row["lon"], row["lat"]))
        if row["st_no"] == st_no_comp:
            G.add_edge(prev_node, row["st_id"], arr_time=convertTimetoMinute(row["arr_time"], row["date"]),
                       dep_time=prev_dep_time, train=row["train"], train_max_speed=getTrainMaxSpeed(row["train"]),
                       day=row["date"])
            st_no_comp = row["st_no"] + 1
            prev_node = row["st_id"]
            prev_mileage = row["mileage"]
            prev_dep_time = convertTimetoMinute(row["dep_time"], row["date"])
        else:
            prev_dep_time = convertTimetoMinute(row["dep_time"], row["date"])
            st_no_comp = row["st_no"] + 1
            prev_node = row["st_id"]
            prev_mileage = row["mileage"]

    return G




def create_network_from_edges(path):

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

    return G


