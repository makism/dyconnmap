# -*- coding: utf-8 -*-
""" Growing NeuralGas

Growing Neural Gas (*GNG*) [Fritzke1995]_ is a dynamic neural network (as in adaptive) that learns topologies.
Compared to Neural Gas, GNG provides the functionality of adding or purging the constructed graph of
nodes and edges when certain criterion are met.

To do so, each node on the network stores a number of secondary information and statistics,
such as, its learning vector, a local error, etc. Edge edge is assigned with a counter related
to its age; so as older edges are pruned.

The convergence of the algorithm depends either on the maximum number of nodes of the graph,
or an upper limit of elapsed iterations.

Briefly, the algorithm works as following:


1. Create two nodes, with weights drawn randomly from the original distibution; connect these two nodes. Set the edge's age to zero.

2. Draw randomly a sample (:math:`\\overrightarrow{x}`) from the distibution.

3. For each node (:math:`n`) in the graph with associated weights :math:`\\overrightarrow{w}`, we compute the euclidean distance from :math:`\\overrightarrow{x}`: :math:`||\\overrightarrow{n}_w - \\overrightarrow{x}||^2`. Next, we find the two nodes closest :math:`\\overrightarrow{x}` with distances :math:`d_s` and :math:`d_t`.

4. The best matching unit (:math:`s`) adjusts:
 a. its weights: :math:`\\overrightarrow{s}_w \\leftarrow \\overrightarrow{s}_w + [e_w * (\\overrightarrow{x} - \\overrightarrow{s}_w)]`.
 b. its local error: :math:`s_{error} \\leftarrow s_{error} + d_s`.

5. Next, the nodes (:math:`N`) adjacent to :math:`s`:
 a. update their weights: :math:`\\overrightarrow{N}_w \\leftarrow \\overrightarrow{N}_w + [e_n * (\\overrightarrow{x} - \\overrightarrow{N}_w)]`.
 b. increase the age of the connecting edges by one.

6. If the best and second mathing units (:math:`s` and :math:`t`) are connected, we reset the age of the connecting edge. Otherwise, we connect them.

7. Regarding the pruning of the network. First we remove the edges with older than :math:`a_{max}`. In the seond pass, we remove any disconnected nodes.

8. We check the iteration (:math:`iter`), whether is a multiple of :math:`\\lambda` and if the maximum number of iteration has been reached; then we add a new node (:math:`q`) in the graph:
 a. Let :math:`u` denote the node with the highest error on the graph, and :math:`v` its neighbor with the highest error.
 b. we disconnect :math:`u` and :math:`v`
 c. :math:`q` is added between :math:`u` and :math:`v`: :math:`\\overrightarrow{q}_w \\leftarrow \\frac{ \\overrightarrow{u}_w + \\overrightarrow{v}_w }{2}`.
 d. connect :math:`q` to :math:`u`, and :math:`q` to :math:`v`
 e. reduce the local errors of both :math:`u` and :math:`v`: :math:`u_{error} \\leftarrow \\alpha * u_{error}` and :math:`v_{error} \\leftarrow \\alpha * v_{error}`
 f. define the local error :math:`q`: :math:`q_{error} \\leftarrow u_{error}`

8. Adjust the error of each node (:math:`n`) on the graph: :math:`n_{error} \\leftarrow n_{error} - \\beta * n_{error}`

9. Finally, increate the iteration (:math:`iter`) and if any of the criterion is not satisfied, repeat from step #2.


|

-----

.. [Fritzke1995] Fritzke, B. (1995). A growing neural gas network learns topologies. In Advances in neural information processing systems (pp. 625-632).

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .cluster import BaseCluster


class GrowingNeuralGas(BaseCluster):
    """ Growing Neural Gas


    Parameters
    ----------
    n_max_protos : int
        Maximum number of nodes.

    l : int
        Every iteration is checked if it is a multiple of this value.

    a_max : int
        Maximum age of edges.

    a : float
        Weights the local error of the nodes when adding a new node.

    b : float
        Weights the local error of all the nodes on the graph.

    iterations : int
        Total number of iterations.

    lrate : list of length 2
        The learning rates of the best matching unit and its neighbors.

    n_jobs : int
        Number of parallel jobs (will be passed to scikit-learn)).

    rng : object or None
        An object of type numpy.random.RandomState.


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
        lrate=None,
        # lrate=[0.05, 0.0006],
        n_jobs=1,
        rng=None,
    ):
        if lrate is None:
            self.ew, self.en = [0.05, 0.0006]
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

        self.metric = "euclidean"

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
        n_samples, _ = np.shape(data)

        initial_indices = self.rng.choice(n_samples, size=2, replace=False)

        g = nx.Graph()
        g.add_node(0, pos=data[initial_indices[0], :], error=0.0)
        g.add_node(1, pos=data[initial_indices[1], :], error=0.0)
        g.add_edge(0, 1, age=0.0)

        for step in range(self.max_iterations):
            sample_indice = self.rng.choice(n_samples, size=1, replace=False)
            sample = data[sample_indice, :]

            # all_pos = {
            # node_id: np.squeeze(datadict["pos"]).tolist()
            # for node_id, datadict in g.nodes(data=True)
            # }

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

                # n1, error = all_nodes[0]
                n1, _ = all_nodes[0]

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
                g.nodes[n1]["error"] *= self.alpha
                g.nodes[neighborhood_with_errors[0][0]]["error"] *= self.alpha
                g.nodes[new_node_id]["error"] = g.nodes[n1]["error"]

            # Global error reduction
            # for node, datadict in list(g.nodes(data=True)):
            for node, _ in list(g.nodes(data=True)):
                g.nodes[node]["error"] -= self.beta * g.nodes[node]["error"]

        self.__g = g
        self.protos = np.squeeze(
            list([datadict["pos"] for node, datadict in g.nodes(data=True)])
        )

        return self
