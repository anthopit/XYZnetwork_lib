from community import community_louvain
from sklearn.cluster import SpectralClustering, KMeans


def get_clusters(graph, type='louvain', embedding=None, k=None):
    if type == 'louvain':
        return community_louvain.best_partition(graph)
    if type== 'greedy':
        communities = nx.community.greedy_modularity_communities(graph)
        comms_dct = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comms_dct[node] = i
    elif type == 'spectral':
        clustering = SpectralClustering(n_clusters=k, assign_labels='discretize', random_state=0).fit(embedding)
        # Get index of the graphs nodes
        node_index = list(graph.nodes())
        comm_dct = dict(zip(node_index, clustering.labels_))
        comm_dct = {k: v + 1 for k, v in comm_dct.items()}
        return comm_dct
    elif type == 'kmeans':
        clustering = KMeans(n_clusters=k,random_state=0).fit(embedding)
        node_index = list(graph.nodes())
        comm_dct = dict(zip(node_index, clustering.labels_))
        comm_dct = {k: v + 1 for k, v in comm_dct.items()}
        return comm_dct
    else:
        raise Exception(f'Unknown community type: {type}')

def map_clusters(TN, clusters_dct):
    map_weighted_network(TN, custom_node_weigth=clusters_dct, edge_weigth=False, node_size=5, discrete_color=True)

def plot_clusters_embedding(embedding, clusters_dct):
    plot_tsne_embedding(embedding, node_cluster=clusters_dct)

def find_optimal_k_elbow(X, max_k, init='k-means++', n_init=10, random_state=None):
    """
    Find the optimal k for k-means clustering using the Elbow method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    max_k : int
        Maximum value of k to be tested.
    init : str, optional, default: 'k-means++'
        Method for initialization, as accepted by KMeans.
    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different centroid seeds.
    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization.

    Returns
    -------
    int
        Optimal value of k.
    """
    X = X.values
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=random_state)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Find the "elbow" point
    x = np.arange(1, max_k + 1)
    y = np.array(sse)
    curvature = np.abs(np.diff(y, 2))  # Second derivative
    k_elbow = np.argmax(curvature) + 1  # Add 1 to match k index

    # Plot the elbow curve
    plt.figure()
    plt.plot(x, y, 'b*-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.plot(k_elbow, sse[k_elbow - 1], 'ro')  # Mark the elbow point
    plt.show()

    return k_elbow


