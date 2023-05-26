import numpy as np
cimport numpy as np

from cython cimport floating
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libc.float cimport DBL_MAX, FLT_MAX

from sklearn.utils.extmath import row_norms
from sklearn.utils._cython_blas cimport _gemm
from sklearn.utils._cython_blas cimport RowMajor, Trans, NoTrans
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.cluster._k_means_common cimport _relocate_empty_clusters_dense
from sklearn.cluster._k_means_common cimport _relocate_empty_clusters_sparse
from sklearn.cluster._k_means_common cimport _average_centers, _center_shift


np.import_array()

def lloyd_iter_chunked_dense_with_min_sample(
        floating[:, ::1] X,                # IN READ-ONLY
        floating[::1] sample_weight,       # IN READ-ONLY
        floating[::1] x_squared_norms,     # IN
        floating[:, ::1] centers_old,      # IN
        floating[:, ::1] centers_new,      # OUT
        floating[::1] weight_in_clusters,  # OUT
        int[::1] labels,                   # OUT
        floating[::1] center_shift,        # OUT
        int[::1] max_index,                # OUT
        floating[::1] max_distance,        # OUT
        int n_threads,
        bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with dense input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,), dtype=floating
        Squared L2 norm of X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    centers_squared_norms : ndarray of shape (n_clusters,), dtype=floating
        Squared L2 norm of the centers.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_new.shape[0]

        # hard-coded number of samples per chunk. Appeared to be close to
        # optimal in all situations.
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx, n_samples_chunk_eff
        int start, end

        int j, k

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk
        floating *pairwise_distances_chunk

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
    max_distances = <floating*> malloc(n_chunks * sizeof(floating))
    max_indices = <int*> malloc(n_chunks * sizeof(int))
    max_index[0] = -1
    max_distance[0] = 0
    with nogil, parallel(num_threads=n_threads):
        # thread local buffers
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
        pairwise_distances_chunk = <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_dense_with_min_sample(
                X[start: end],
                sample_weight[start: end],
                x_squared_norms[start: end],
                centers_old,
                centers_squared_norms,
                labels[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                pairwise_distances_chunk,
                max_indices+chunk_idx,
                max_distances+chunk_idx,
                update_centers)

        # reduction from local buffers. The gil is necessary for that to avoid
        # race conditions.

        #with gil:
        #    if maxdistances[0] > max_distance[0]:
        #        max_index[0] = maxindices[0]+chunk_idx * n_samples_chunk
        #        max_distance[0] = maxdistances[0]


        if update_centers:
            with gil:
                for j in range(n_clusters):
                    weight_in_clusters[j] += weight_in_clusters_chunk[j]
                    for k in range(n_features):
                        centers_new[j, k] += centers_new_chunk[j * n_features + k]

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)
        free(pairwise_distances_chunk)

    max_index[0] = -1
    max_distance[0] = 0

    for chunk_idx in range(n_chunks):
        if max_distances[chunk_idx] > max_distance[0]:
            max_index[0] = max_indices[chunk_idx]+chunk_idx * n_samples_chunk
            max_distance[0] = max_distances[chunk_idx]
    free(max_indices)
    free(max_distances)
    if update_centers:
        _relocate_empty_clusters_dense(X, sample_weight, centers_old,
                                    centers_new, weight_in_clusters, labels)

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_dense_with_min_sample(
        floating[:, ::1] X,                   # IN READ-ONLY
        floating[::1] sample_weight,          # IN READ-ONLY
        floating[::1] x_squared_norms,        # IN
        floating[:, ::1] centers_old,         # IN
        floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                      # OUT
        floating *centers_new,                # OUT
        floating *weight_in_clusters,         # OUT
        floating *pairwise_distances,         # OUT
        int* max_index,                   # OUT
        floating* max_distance,           # OUT
        bint update_centers) nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, &X[0, 0], n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    max_index[0] = -1
    max_distance[0] = 0
    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        if min_sq_dist+x_squared_norms[i:i+1][0] > max_distance[0]:
            max_index[0] = i
            max_distance[0] = min_sq_dist+x_squared_norms[i:i+1][0]
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i, k] * sample_weight[i]


def update_centers(
    floating[:, ::1] X,                   # IN READ-ONLY
    floating[::1] sample_weight,          # IN READ-ONLY
    int[::1] labels,                      # IN
    floating[:, ::1] centers_new,      # OUT
    floating[::1] weight_in_clusters):  # OUT
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_new.shape[0]
        int k, label
    memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
    memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))


    for i in range(n_samples):
        label = labels[i]
        weight_in_clusters[label] += sample_weight[i]
        for k in range(n_features):
            centers_new[label,k] += X[i, k] * sample_weight[i]
