from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

import numpy as np


def prepare_data(config, results):

    # Step 1 - generate data and y
    n_points = data.shape[0]

    # Step 2
    metric = config.TSNE.METRIC
    perplexity = config.TSNE.PERPLEXITY
    # TODO: transfer to GPU tensor
    distances2 = pairwise_distances(data, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij2 = squareform(pij)

    i, j = np.indices(pij2.shape)
    i, j = i.ravel(), j.ravel()
    out_p_ij = pij2.ravel().astype('float32')
    # remove self-indices
    idx = i != j
    i, j, out_p_ij = i[idx], j[idx], out_p_ij[idx]
    return n_points, pij, i, j, y