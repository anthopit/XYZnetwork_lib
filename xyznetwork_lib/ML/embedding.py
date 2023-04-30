import pandas as pd
from node2vec import Node2Vec as n2v
from visualisation.visualisation import *


class Embedding():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._cached_embedding = None

    def get_embedding(self, graph):
        raise NotImplementedError

    def get_embedding_df(self, graph):
        if self._cached_embedding is None:
            emb = self.get_embedding(graph)
        else:
            emb = self._cached_embedding

        if isinstance(emb, np.ndarray):
            emb_df = pd.DataFrame(emb, index=graph.nodes)
        else:
            emb_df = (
                pd.DataFrame(
                    [emb.get_vector(str(n)) for n in graph.nodes()],
                    index=graph.nodes
                )
            )

        return emb_df

    def plot_embedding(self, graph, node_cluster=None):
        emb_df = self.get_embedding_df(graph)
        plot_tsne_embedding(emb_df)

class GraphWave(Embedding):
    def __init__(self):
        super().__init__()

    def get_embedding(self, graph):
        chi, heat_print, taus = graphwave_alg(graph, np.linspace(0, 100, 25), taus='auto', verbose=True)

        return chi

class Node2Vec(Embedding):
    def __init__(self, window_size=10, min_count=1, batch_word=4, emb_size=64, walk_length=4, num_walks=100, weight_key=None, workers=1, p=1, q=0.5):
        super().__init__()
        self.window_size = window_size
        self.min_count = min_count
        self.batch_word = batch_word
        self.emb_size = emb_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.weight_key = weight_key
        self.workers = workers
        self.p = p
        self.q = q

    def get_embedding(self, graph):
        if self._cached_embedding is not None:
            return self._cached_embedding

        g_emb_struct = n2v(
            graph,
            dimensions=self.emb_size,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            weight_key=self.weight_key,
            workers=self.workers,
            p=self.p,
            q=self.q,
        )
        mdl_struct = g_emb_struct.fit(vector_size=self.emb_size,
                                 window=self.window_size,
                                 min_count=self.min_count,
                                 batch_words=self.batch_word,)
        emb = mdl_struct.wv
        self._cached_embedding = emb
        return emb


class LaplacianEigenmaps(Embedding):
    def __init__(self, emb_size=3):
        super().__init__()
        self.emb_size = emb_size

    def get_embedding(self, graph):
        # Create the adjacency matrix
        A = nx.to_numpy_array(graph)

        # Compute the degree matrix
        D = np.diag(np.sum(A, axis=1))

        # Compute the Laplacian matrix
        L = D - A

        # Compute the eigenvectors and eigenvalues of L
        eigenvals, eigenvecs = np.linalg.eig(L)

        # Sort the eigenvectors by their corresponding eigenvalues
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Select the k eigenvectors corresponding to the k smallest eigenvalues
        k = self.emb_size
        X = eigenvecs[:, :k]

        # Normalize the rows of X
        X_norm = np.linalg.norm(X, axis=1)
        X_norm[X_norm == 0] = 1
        X = X / X_norm[:, None]

        return X

    def plot_embedding(self, graph, node_cluster=None):
        print("Laplacian Eigenmaps does not support plotting")

class AdjencyMatrix(Embedding):
    def __init__(self):
        super().__init__()

    def get_embedding(self, graph):
        # Create the adjacency matrix
        A = nx.to_numpy_array(graph)

        return A

    def plot_embedding(self, graph, node_cluster=None):
        if self._cached_embedding is None:
            emb = self.get_embedding(graph)
        else:
            emb = self._cached_embedding

        plt.matshow(emb)

        plt.show()