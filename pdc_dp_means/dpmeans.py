from time import time


import numpy as np
import scipy.sparse as sp

from sklearn.cluster._k_means_common import _inertia_dense
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
from sklearn.cluster._kmeans import (
    KMeans,
    _labels_inertia_threadpool_limit,
    _minibatch_update_dense,
)

from sklearn.utils import check_array, check_random_state, deprecated
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import threadpool_limits
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from .dp_means_cython import lloyd_iter_chunked_dense_with_min_sample


def _dpmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
    delta=1.0,
    max_clusters=None,
):
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)
    max_index = np.full(1, -1, dtype=np.int32)
    max_distance = np.full(1, -1, dtype=centers.dtype)

    lloyd_iter = lloyd_iter_chunked_dense_with_min_sample
    lloyd_kmeans_iter = lloyd_iter_chunked_dense
    _inertia = _inertia_dense

    strict_convergence = False
    iter_time = []
    # all_centers = [None]
    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            tic = time()
            lloyd_iter(
                X,
                sample_weight,
                x_squared_norms,
                centers,
                centers_new,
                weight_in_clusters,
                labels,
                center_shift,
                max_index,
                max_distance,
                n_threads,
                update_centers=True,
            )

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                print(f"Iteration {i}, inertia {inertia}.")
            new_cluster = False

            if max_clusters is None or max_clusters > centers.shape[0]:
                if max_index[0] != -1 and max_distance[0] > delta:
                    centers = np.vstack((centers, X[max_index])).astype(X.dtype)
                    centers_new = np.vstack((centers_new, X[max_index])).astype(
                        X.dtype
                    )
                    weight_in_clusters = np.hstack([weight_in_clusters, [0]]).astype(
                        X.dtype
                    )
                    center_shift = np.hstack([center_shift, [0]]).astype(X.dtype)
                    new_cluster = True

            # update_centers(X,sample_weight,labels,centers_new,weight_in_clusters)

            centers, centers_new = centers_new, centers
            toc = time()
            iter_time.append(toc - tic)
            # all_centers.append(centers)

            if new_cluster is False:
                if np.array_equal(labels, labels_old):
                    # First check the labels for strict convergence.
                    if verbose:
                        print(f"Converged at iteration {i}: strict convergence.")
                    strict_convergence = True
                    break
                else:
                    # No strict convergence, check for tol based convergence.
                    center_shift_tot = (center_shift**2).sum()
                    if center_shift_tot <= tol:
                        if verbose:
                            print(
                                f"Converged at iteration {i}: center shift "
                                f"{center_shift_tot} within tolerance {tol}."
                            )
                        break

            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_kmeans_iter(
                X,
                sample_weight,
                centers,
                centers,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
                update_centers=False,
            )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1, 0, iter_time


class DPMeans(KMeans):
    """DP-Means clustering.

    The DP-Means is an extension of the K-Means algorithm inspired by the Dirichlet Process Mixture model.
    This allows the number of clusters to be learned from the data, instead of being set beforehand.

    Parameters
    ----------

    n_clusters : int, default=8
        The initial number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization. Same as KMeans initialization.

    max_iter : int, default=300
        Maximum number of iterations of the DP-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first.

    delta : float, default=1.0
        The parameter controls the balance between the number of clusters and the data fitting term.
        Higher values of delta would generate fewer clusters, lower values would generate more clusters.

    max_clusters : int or None, default=None
        The maximum number of clusters that can be formed. Useful for controlling runtime
        in the case where it's suspected that delta is set too low.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : The base algorithm for DP-Means. Fixed number of clusters.

    Notes
    -----
    The DP-Means algorithm extends K-Means by treating the number of clusters as a variable to be learned.
    A new cluster is formed whenever a data point is "far enough" from all existing clusters.
    "Far enough" is determined by the `delta` parameter, which effectively controls the number of clusters formed.


    Examples
    --------

    >>> from pdc_dp_means import DPMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> dpmeans = DPMeans(n_clusters=2, delta=1.0, random_state=0).fit(X)
    >>> dpmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> dpmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> dpmeans.cluster_centers_
    array([[10.,  2.],
        [ 1.,  2.]])
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        delta=1.0,
        max_clusters=None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
            copy_x=copy_x,
        )
        self.delta = delta
        self.max_clusters = max_clusters

    def fit(self, X, y=None, sample_weight=None):
        """Compute dp-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        super()._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        dpmeans_single = _dpmeans_single_lloyd
        self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
            )

            if self.verbose:
                print("Initialization complete")

            # run a dp-means once

            labels, inertia, centers, n_iter_, all_centers, iter_time = dpmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                x_squared_norms=x_squared_norms,
                n_threads=self._n_threads,
                delta=self.delta,
                max_clusters=self.max_clusters,
            )

            inertia += self.delta * centers.shape[0]

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (inertia < best_inertia):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean
        self.n_clusters = best_centers.shape[0]
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        # self.iter_time = iter_time
        # self.all_centers = all_centers
        return self


def _labels_inertia_with_min_sample(
    X, sample_weight, x_squared_norms, centers, n_threads=1
):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    max_index = np.full(1, -1, dtype=np.int32)
    max_distance = np.full(1, -1, dtype=centers.dtype)
    weight_in_clusters = np.zeros(n_clusters, dtype=centers.dtype)
    center_shift = np.zeros_like(weight_in_clusters)

    _labels = lloyd_iter_chunked_dense_with_min_sample
    _inertia = _inertia_dense
    X = X

    _labels(
        X,
        sample_weight,
        x_squared_norms,
        centers,
        centers,
        weight_in_clusters,
        labels,
        center_shift,
        max_index,
        max_distance,
        n_threads,
        update_centers=False,
    )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, max_index, max_distance


def _mini_batch_step_with_max_distance(
    X,
    x_squared_norms,
    sample_weight,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. If sparse, must be in CSR format.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared euclidean norm of each data point.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    # Perform label assignment to nearest centers
    # For better efficiency, it's better to run _mini_batch_step in a
    # threadpool_limit context than using _labels_inertia_threadpool_limit here
    labels, inertia, max_index, max_distance = _labels_inertia_with_min_sample(
        X, sample_weight, x_squared_norms, centers, n_threads=n_threads
    )

    # Update centers according to the labels

    _minibatch_update_dense(
        X,
        sample_weight,
        centers,
        centers_new,
        weight_sums,
        labels,
        n_threads,
    )

    # Reassign clusters that have very low weight

    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]) :]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()

        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(
                X.shape[0], replace=False, size=n_reassigns
            )
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

            if sp.issparse(X):
                assign_rows_csr(
                    X,
                    new_centers.astype(np.intp, copy=False),
                    np.where(to_reassign)[0].astype(np.intp, copy=False),
                    centers_new,
                )
            else:
                centers_new[to_reassign] = X[new_centers]

        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    return inertia, max_index[0], max_distance[0]


class MiniBatchDPMeans(KMeans):
    """
    Parameters
    ----------

    n_clusters : int, default=1
        The initial number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids for deterministic 
        initialization using sampling based on an empirical probability distribution 
        of the points' contribution to the overall inertia. This technique 
        speeds up convergence. The algorithm implemented is "greedy k-means++". 
        It differs from the vanilla k-means++ by making several trials at each 
        sampling step and choosing the best centroid among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes.

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch DPMeans on a
        random subset of the data. This needs to be larger than n_clusters.

    n_init : int, default=3
        Number of random initializations that are tried.
        The algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering.

    delta : float, default=1.0
        Parameter controlling the number of clusters in the DP-means
        algorithm. A higher value will lead to fewer clusters.

    Attributes
    ----------
        cluster_centers_ : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers.

        labels_ : ndarray of shape (n_samples,)
            Labels of each point (if compute_labels is set to True).

        inertia_ : float
            The value of the inertia criterion associated with the chosen
            partition if compute_labels is set to True. If compute_labels is set to
            False, it's an approximation of the inertia based on an exponentially
            weighted average of the batch inertiae.
            The inertia is defined as the sum of square distances of samples to
            their cluster center, weighted by the sample weights if provided.

        n_iter_ : int
            Number of iterations over the full dataset.

        n_features_in_ : int
            Number of features seen during :term:`fit`.

        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.

        See Also
        --------
        DPMeans : The full batch version of DP-Means clustering.

        KMeans : The classic implementation of the clustering method based on the
            Lloyd's algorithm. It consumes the whole set of input data at each
            iteration.

        Notes
        -----
        When there are too few points in the dataset, some centers may be
        duplicated, which means that a proper clustering in terms of the number
        of requesting clusters and the number of returned clusters will not
        always match. One solution is to set `reassignment_ratio=0`, which
        prevents reassignments of clusters that are too small.

        Examples
        --------
        >>> from pdc_dp_means import MiniBatchDPMeans
        >>> import numpy as np
        >>> X = np.array([[1, 2], [1, 4], [1, 0],
        ...               [4, 2], [4, 0], [4, 4],
        ...               [4, 5], [0, 1], [2, 2],
        ...               [3, 2], [5, 5], [1, -1]])
        >>> # manually fit on batches
        >>> dpmeans = MiniBatchDPMeans(n_clusters=1,
        ...                            random_state=0,
        ...                            batch_size=6,
        ...                            n_init=3,
        ...                            delta=1.0)
        >>> dpmeans = dpmeans.partial_fit(X[0:6,:])
        >>> dpmeans = dpmeans.partial_fit(X[6:12,:])
        >>> dpmeans.cluster_centers_
        array([[3.375, 3.  ],
            [0.75 , 0.5 ]])
        >>> dpmeans.predict([[0, 0], [4, 4]])
        array([1, 0], dtype=int32)
        >>> # fit on the whole data
        >>> dpmeans = MiniBatchDPMeans(n_clusters=1,
        ...                            random_state=0,
        ...                            batch_size=6,
        ...                            max_iter=10,
        ...                            n_init=3,
        ...                            delta=1.0).fit(X)
        >>> dpmeans.cluster_centers_
        array([[3.55102041, 2.48979592],
            [1.06896552, 1.        ]])
        >>> dpmeans.predict([[0, 0], [4, 4]])
        array([1, 0], dtype=int32)

    
    """

    def __init__(
        self,
        n_clusters=1,
        *,
        init="k-means++",
        max_iter=100,
        batch_size=1024,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
        delta=1.0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio
        self.delta = delta

    @deprecated(  # type: ignore
        "The attribute `counts_` is deprecated in 0.24"
        " and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def counts_(self):
        return self._counts

    @deprecated(  # type: ignore
        "The attribute `init_size_` is deprecated in "
        "0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def init_size_(self):
        return self._init_size

    @deprecated(  # type: ignore
        "The attribute `random_state_` is deprecated "
        "in 0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def random_state_(self):
        return getattr(self, "_random_state", None)

    def _check_params(self, X):
        super()._check_params_vs_input(X)

        # max_no_improvement
        if self.max_no_improvement is not None and self.max_no_improvement < 0:
            raise ValueError(
                "max_no_improvement should be >= 0, got "
                f"{self.max_no_improvement} instead."
            )

        # batch_size
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be > 0, got {self.batch_size} instead."
            )
        self._batch_size = min(self.batch_size, X.shape[0])

        # init_size
        if self.init_size is not None and self.init_size <= 0:
            raise ValueError(f"init_size should be > 0, got {self.init_size} instead.")
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        # reassignment_ratio
        if self.reassignment_ratio < 0:
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        """Helper function to encapsulate the early stopping logic"""
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self._batch_size

        # count steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because it's inertia from initialization.
        if step == 1:
            if self.verbose:
                print(
                    f"Minibatch step {step}/{n_steps}: mean batch "
                    f"inertia: {batch_inertia}"
                )
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        # Early stopping based on absolute tolerance on squared change of
        # centers position
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in inertia) at step "
                    f"{step}/{n_steps}"
                )
            return True

        return False

    def _random_reassign(self):
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False

    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        self._check_mkl_vcomp(X, self._batch_size)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # Validation set for the init
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]

        # perform several inits with random subsets
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            cluster_centers = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                init_size=self._init_size,
            )

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid,
                sample_weight_valid,
                cluster_centers,
                n_threads=self._n_threads,
            )

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        # Initialize number of samples seen since last reassignment
        self._n_since_last_reassign = 0

        n_steps = (self.max_iter * n_samples) // self._batch_size

        iter_time = [0]
        self.iter_centers = [None]

        with threadpool_limits(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_steps):
                # Sample a minibatch from the full dataset
                minibatch_indices = random_state.randint(
                    0, n_samples, self._batch_size
                )
                tic = time()
                # Perform the actual update step on the minibatch data
                (
                    batch_inertia,
                    max_index,
                    max_distance,
                ) = _mini_batch_step_with_max_distance(
                    X=X[minibatch_indices],
                    x_squared_norms=x_squared_norms[minibatch_indices],
                    sample_weight=sample_weight[minibatch_indices],
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads,
                )
                toc = time() - tic
                iter_time.append(toc)
                new_cluster = False
                if max_index != -1 and max_distance > self.delta:
                    centers = np.vstack(
                        (centers, X[minibatch_indices[max_index]])
                    ).astype(X.dtype)
                    centers_new = np.vstack(
                        (centers_new, X[minibatch_indices[max_index]])
                    ).astype(X.dtype)
                    self.n_clusters += 1
                    self._counts = np.hstack([self._counts, [1]]).astype(X.dtype)
                    new_cluster = True

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0
                centers, centers_new = centers_new, centers
                self.iter_centers.append(centers)
                # Monitor convergence and do early stopping if necessary
                if new_cluster is False and self._mini_batch_convergence(
                    i, n_steps, n_samples, centers_squared_diff, batch_inertia
                ):
                    break

        self.cluster_centers_ = centers
        self.iter_time = iter_time
        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil(((i + 1) * self._batch_size) / n_samples))

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
            self.inertia_ += self.delta * self.cluster_centers_.shape[0]
        else:
            self.inertia_ = (
                self._ewa_inertia * n_samples
                + self.delta * self.cluster_centers_.shape[0]
            )

        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Return updated estimator.
        """
        has_centers = hasattr(self, "cluster_centers_")

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=not has_centers,
        )

        self._random_state = getattr(
            self, "_random_state", check_random_state(self.random_state)
        )
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, "n_steps_", 0)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if not has_centers:
            # this instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            self._n_threads = _openmp_effective_n_threads()

            # Validate init array
            init = self.init
            if hasattr(init, "__array__"):
                init = check_array(init, dtype=X.dtype, copy=True, order="C")
                self._validate_center_shape(X, init)

            self._check_mkl_vcomp(X, X.shape[0])

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=self._random_state,
                init_size=self._init_size,
            )

            # Initialize counts
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

            # Initialize number of samples seen since last reassignment
            self._n_since_last_reassign = 0

        with threadpool_limits(limits=1, user_api="blas"):
            _, max_index, max_distance = _mini_batch_step_with_max_distance(
                X,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
                centers=self.cluster_centers_,
                centers_new=self.cluster_centers_,
                weight_sums=self._counts,
                random_state=self._random_state,
                random_reassign=self._random_reassign(),
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose,
                n_threads=self._n_threads,
            )

            if max_index != -1 and max_distance > self.delta:
                self.cluster_centers_ = np.vstack(
                    (self.cluster_centers_, X[max_index])
                ).astype(X.dtype)
                self.n_clusters += 1
                self._counts = np.hstack([self._counts, [1]]).astype(X.dtype)

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
            self.inertia_ += self.delta * self.cluster_centers_.shape[0]

        self.n_steps_ += 1

        return self

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        labels, _ = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            self.cluster_centers_,
            n_threads=self._n_threads,
        )

        return labels

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
