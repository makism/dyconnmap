# -*- coding: utf-8 -*-
""" Growing NeuralGas


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS


class GrowingNeuralGas:
    """ Growing Neural Gas


    Parameters
    ----------

    n_jobs : int
        Number of parallel jobs (will be passed to scikit-learn))

    rng : object or None
        An object of type numpy.random.RandomState


    Attributes
    ----------
    protos : array-like, shape(n_protos, n_features)
        The prototypical vectors

    Notes
    -----

    """

    def __init__(
        self,
        n_max_protos=30,
        l=300,
        a_max=88,
        a=0.5,
        b=0.0005,
        iterations=10000,
        lrate=[0.05, 0.0006],
        n_jobs=1,
        rng=None,
    ):
        self.ew, self.en = lrate
        self.a_max = a_max
        self.total_lambda = l
        self.max_nodes = n_max_protos
        self.alpha = a
        self.beta = b
        self.max_iterations = iterations
        self.protos = None
        self.n_jobs = n_jobs

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.__nodes = None
        self.__g = nx.Graph()

        self.__symbols = None
        self.__encoding = None

    def fit(self, data):
        """ Learn data, and construct a vector codebook.

        Parameters
        ----------
        data : real array-like, shape(n_samples, n_features)
            Data matrix, each row represents a sample.

        Returns
        -------
        self : object
            The instance itself
        """
        n_samples, n_features = np.shape(data)

        initial_indices = self.rng.choice(n_samples, size=2, replace=False)

        g = nx.Graph()
        g.add_node(0, pos=data[initial_indices[0], :], error=0.0)
        g.add_node(1, pos=data[initial_indices[1], :], error=0.0)
        g.add_edge(0, 1, age=0.0)

        for step in range(self.max_iterations):
            sample_indice = self.rng.choice(n_samples, size=1, replace=False)
            sample = data[sample_indice, :]

            all_pos = {
                node_id: np.squeeze(datadict["pos"]).tolist()
                for node_id, datadict in g.nodes(data=True)
            }

            # Get nodes from the Graph
            nodes = [
                (node_id, datadict["pos"])
                for node_id, datadict in list(g.nodes(data=True))
            ]

            positions = [pos for _, pos in nodes]
            positions = np.squeeze(positions)

            # Compute distances
            D = pairwise_distances(
                sample, positions, metric="euclidean", n_jobs=self.n_jobs
            )
            D = np.squeeze(D)
            I = np.argsort(D)
            I = np.squeeze(I)

            # Find two closest nodes
            bmu1, bmu2 = I[0:2]

            # Update bmu's error
            g.nodes[bmu1]["error"] += D[bmu1]

            # Update bmu's position
            bmu_pos = g.nodes[bmu1]["pos"]
            g.nodes[bmu1]["pos"] = bmu_pos + (sample - bmu_pos) * self.ew

            # Bmu's neighborhood
            nbrs = g.adj[bmu1].items()

            for nbr in nbrs:
                # Adjust positions
                nbr_pos = g.nodes[nbr[0]]["pos"]
                g.nodes[nbr[0]]["pos"] = nbr_pos + ((sample - nbr_pos) * self.en)

                # Adjust edges' age
                g.edges[bmu1, nbr[0]]["age"] += 1.0

            # Connect bmu1 and bmu2 if needed
            if not g.has_edge(bmu1, bmu2):
                g.add_edge(bmu1, bmu2, age=0.0)
            else:
                g.edges[bmu1, bmu2]["age"] = 0.0

            # Delete old connections
            old_edges = [
                (u, v)
                for u, v, datadict in g.edges(data=True)
                if datadict["age"] > self.a_max
            ]
            g.remove_edges_from(old_edges)

            # Delete isolated nodes
            g.remove_nodes_from(list(nx.isolates(g)))

            # New node insertion
            num_nodes = g.number_of_nodes()
            if step % self.total_lambda == 0 and num_nodes < self.max_nodes:
                all_nodes = [
                    (n, datadict["error"]) for n, datadict in g.nodes(data=True)
                ]
                all_nodes = sorted(
                    all_nodes, key=lambda nodedata: nodedata[1], reverse=True
                )

                n1, error = all_nodes[0]

                neighborhood = list(g.neighbors(n1))

                neighborhood_with_errors = list(
                    filter(lambda nodedata: nodedata[0] in neighborhood, all_nodes)
                )
                neighborhood_with_errors = sorted(
                    neighborhood_with_errors,
                    key=lambda nodedata: nodedata[1],
                    reverse=True,
                )

                node1 = g.nodes[n1]
                node2 = g.nodes[neighborhood_with_errors[0][0]]

                new_node_id = g.number_of_nodes()
                new_node_pos = (node1["pos"] + node2["pos"]) / 2.0
                g.add_node(new_node_id, pos=new_node_pos, error=0.0)

                # Disconnect the two nodes
                g.remove_edge(n1, neighborhood_with_errors[0][0])
                # Connect the new nodes with the other two ones
                g.add_edge(n1, new_node_id, age=0.0)
                g.add_edge(new_node_id, neighborhood_with_errors[0][0], age=0.0)

                # Update the errors
                g.node[n1]["error"] *= self.alpha
                g.node[neighborhood_with_errors[0][0]]["error"] *= self.alpha
                g.node[new_node_id]["error"] = g.node[n1]["error"]

            # Global error reduction
            for node, datadict in list(g.nodes(data=True)):
                g.nodes[node]["error"] -= self.beta * g.nodes[node]["error"]

        self.__g = g
        self.protos = list([datadict["pos"] for node, datadict in g.nodes(data=True)])

        return self

    def encode(self, data, metric=None):
        """

        """
        mds = MDS(1, random_state=self.rng)
        protos_1d = mds.fit_transform(self.protos).ravel()
        sorted_protos_1d = np.argsort(protos_1d)

        sprotos = self.protos[sorted_protos_1d]

        if metric is None:
            metric = self.metric

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto", metric=metric).fit(
            sprotos
        )
        _, self.__symbols = nbrs.kneighbors(data)
        self.__encoding = sprotos[self.__symbols]

        return (self.__encoding, self.__symbols)
