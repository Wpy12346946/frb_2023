"""
Class for construction of a K nearest neighbors index with support for custom distance metrics and
shared nearest neighbors (SNN) distance.

USAGE:
```
from knn_index import KNNIndex

index = KNNIndex(data, **kwargs)
nn_indices, nn_distances = index.query(data_test, k=5)

```
"""
import numpy as np
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from helpers.metrics_custom import (
    distance_SNN,
    remove_self_neighbors
)
from helpers.utils import get_num_jobs
from helpers.constants import (
    NEIGHBORHOOD_CONST,
    MIN_N_NEIGHBORS,
    RHO,
    SEED_DEFAULT,
    METRIC_DEF
)
import warnings
from numba import NumbaPendingDeprecationWarning

# Suppress numba warnings
warnings.filterwarnings('ignore', '', NumbaPendingDeprecationWarning)


def helper_knn_distance(indices1, indices2, distances2):
    """
    :param indices1: integer numpy array of sample indices of shape `(n1, )`
    :param indices2: integer numpy array of sample indices of shape `(n2, )`
    :param distances2: float numpy array of `k` nearest neighbor distances. Has shape `(n2, k)`

    :return: numpy array of `k` nearest neighbor distance corresponding to `indices1`. Has shape `(n1, k)`.
             Distances are set to -1 when they are not available in `distances2`.
    """
    n, k = distances2.shape
    ind_dist_map = {indices2[j]: distances2[j, :] for j in range(n)}
    val_def = -1 * np.ones(k)
    return np.array([ind_dist_map.get(i, val_def) for i in indices1])


class KNNIndex:
    """
    Class for construction of a K nearest neighbors index with support for custom distance metrics and
    shared nearest neighbors (SNN) distance.
    """
    def __init__(self, data,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric=METRIC_DEF, metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 low_memory=False,
                 seed_rng=SEED_DEFAULT):
        """
        :param data: numpy array with the data samples. Has shape `(N, d)`, where `N` is the number of samples and
                     `d` is the number of features.
        :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a
                                      function of the number of samples (data size). If `N` is the number of samples,
                                      then the number of neighbors is set to `N^neighborhood_constant`. It is
                                      recommended to set this value in the range 0.4 to 0.5.
        :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                            the `neighborhood_constant` is ignored. It is sufficient to specify either
                            `neighborhood_constant` or `n_neighbors`.
        :param metric: string or a callable that specifies the distance metric.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary.
        :param shared_nearest_neighbors: Set to True in order to use the shared nearest neighbor (SNN) distance.
                                         This is a secondary distance metric that is found to be better suited to
                                         high dimensional data.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. This is recommended when the number of points is
                                         large and/or when the dimension of the data is high.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param seed_rng: int value specifying the seed for the random number generator.
        """
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        N, d = data.shape
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        # Number of neighbors to use for calculating the shared nearest neighbor distance
        self.n_neighbors_snn = min(int(1.2 * self.n_neighbors), N - 1)
        # self.n_neighbors_snn = self.n_neighbors

        self.nn_indices = None
        self.nn_distances = None
        self.index_knn = self.build_knn_index(data)

    def build_knn_index(self, data, min_n_neighbors=MIN_N_NEIGHBORS, rho=RHO):
        """
        Build a KNN index for the given data set. There will two KNN indices of the SNN distance is used.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param min_n_neighbors: minimum number of nearest neighbors to use for the `NN-descent` method.
        :param rho: `rho` parameter used by the `NN-descent` method.

        :return: A list with one or two KNN indices.
        """
        # Add one extra neighbor because querying on the points that are part of the KNN index will result in
        # the neighbor set containing the queried point. This can be removed from the query result
        if self.shared_nearest_neighbors:
            k = max(1 + self.n_neighbors_snn, min_n_neighbors)
        else:
            k = max(1 + self.n_neighbors, min_n_neighbors)

        # KNN index based on the primary distance metric
        if self.approx_nearest_neighbors:
            params = {
                'metric': self.metric,
                'metric_kwds': self.metric_kwargs,
                'n_neighbors': k,
                # 'rho': rho,
                'random_state': self.seed_rng,
                'n_jobs': self.n_jobs,
                'low_memory': self.low_memory
            }
            index_knn_primary = NNDescent(data, **params)

            self.nn_indices, self.nn_distances = remove_self_neighbors(index_knn_primary._neighbor_graph[0],index_knn_primary._neighbor_graph[1])
        else:
            # Exact KNN graph
            index_knn_primary = NearestNeighbors(
                n_neighbors=k,
                algorithm='brute',
                metric=self.metric,
                metric_params=self.metric_kwargs,
                n_jobs=self.n_jobs
            )
            index_knn_primary.fit(data)

            self.nn_indices, self.nn_distances = remove_self_neighbors(
                *self._query(data, index_knn_primary, k)
            )

        if self.shared_nearest_neighbors:
            # Construct a second KNN index that uses the shared nearest neighbor distance
            data_neighbors = self.nn_indices[:, 0:self.n_neighbors_snn]
            if self.approx_nearest_neighbors:
                params = {
                    'metric': distance_SNN,
                    'n_neighbors': max(1 + self.n_neighbors, min_n_neighbors),
                    'rho': rho,
                    'random_state': self.seed_rng,
                    'n_jobs': self.n_jobs,
                    'low_memory': self.low_memory
                }
                index_knn_secondary = NNDescent(data_neighbors, **params)

                # Save the nearest neighbor information of the data used to build the KNN index
                self.nn_indices, self.nn_distances = remove_self_neighbors(index_knn_secondary._neighbor_graph[0],
                                                                           index_knn_secondary._neighbor_graph[1])
            else:
                index_knn_secondary = NearestNeighbors(
                    n_neighbors=(1 + self.n_neighbors),
                    algorithm='brute',
                    metric=distance_SNN,
                    n_jobs=self.n_jobs
                )
                index_knn_secondary.fit(data_neighbors)

                # Save the nearest neighbor information of the data used to build the KNN index
                self.nn_indices, self.nn_distances = remove_self_neighbors(
                    *self._query(data_neighbors, index_knn_secondary, 1 + self.n_neighbors)
                )

            index_knn = [index_knn_primary, index_knn_secondary]
        else:
            index_knn = [index_knn_primary]

        return index_knn

    def query_self(self, rows=None, k=None):
        """
        Query the nearest neighbors of the points used to construct the KNN index. The index of the points whose
        neighbors are required can be specified via the input `rows`. If this is `None`, then the neighbors of all
        the points are returned.

        :param rows: None or an iterable with the row indices of the points whose neighbors are required.
        :param k: None or an int value specifying the number of neighbors. If `None`, the number of neighbors
                  specified while constructing the graph is used.
        :return: Same as the method `query`.
        """
        if k is None:
            k = self.n_neighbors

        if rows is None:
            return self.nn_indices[:, :k], self.nn_distances[:, :k]
        else:
            return self.nn_indices[rows, :k], self.nn_distances[rows, :k]

    def query(self, data, k=None):
        """
        Query for the `k` nearest neighbors of each point in `data`.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param k: number of nearest neighbors to query. If not specified or set to `None`, `k` will be
                  set to `self.n_neighbors`.

        :return: (nn_indices, nn_distances), where
            - nn_indices: numpy array of indices of the nearest neighbors. Has shape `(data.shape[0], k)`.
            - nn_distances: numpy array of distances of the nearest neighbors. Has shape `(data.shape[0], k)`.
        """
        if k is None:
            k = self.n_neighbors

        if self.shared_nearest_neighbors:
            data_neighbors, _ = self._query(data, self.index_knn[0], self.n_neighbors_snn)
            return self._query(data_neighbors, self.index_knn[1], k)
        else:
            return self._query(data, self.index_knn[0], k)

    def _query(self, data, index, k):
        """
        Unified wrapper for querying both the approximate and the exact KNN index. Do not directly call this
        method unless you have a specific reason.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param index: KNN index.
        :param k: number of nearest neighbors to query.

        :return: (nn_indices, nn_distances), where
            - nn_indices: numpy array of indices of the nearest neighbors. Has shape `(data.shape[0], k)`.
            - nn_distances: numpy array of distances of the nearest neighbors. Has shape `(data.shape[0], k)`.
        """
        if self.approx_nearest_neighbors:
            nn_indices, nn_distances = index.query(data, k=k)
        else:
            nn_distances, nn_indices = index.kneighbors(data, n_neighbors=k)

        return nn_indices, nn_distances
