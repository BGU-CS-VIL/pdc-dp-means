import numpy as np
from sklearn.datasets import make_blobs
from pdc_dp_means import DPMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score


def test_dpmeans():
    # Generate some random data
    X, y = make_blobs(n_samples=1000, centers=6, random_state=42)

    # Create a DPMeans object and fit the data
    dpmeans = DPMeans(n_clusters=1, delta=25, random_state=42)
    labels = dpmeans.fit(X)

    # Check that the number of clusters is correct
    assert dpmeans.n_clusters == 6

    # Check that the cluster centers are close to the true centers
    true_centers = np.array([X[y == i].mean(axis=0) for i in range(6)])
    # Compute the pairwise distances between the true centers and the estimated centers
    distances = cdist(true_centers, dpmeans.cluster_centers_)

    # Find the minimum distance between each true center and any estimated center
    min_distances = np.min(distances, axis=1)

    # Assert that the minimum distance is less than or equal to the tolerance
    assert np.all(min_distances <= 0.5)

    # Compute the NMI between the true labels and the labels computed by DPMeans
    nmi = normalized_mutual_info_score(y, dpmeans.labels_)
    # Assert that the NMI is greater than or equal to a certain threshold (e.g., 0.9)
    assert nmi >= 0.9

    # Check that the inertia is correct
    true_inertia = np.sum((X - true_centers[y]) ** 2)
    assert np.isclose(dpmeans.inertia_, true_inertia, atol=15)

    # Check that the fit method returns self
    assert dpmeans.fit(X) is dpmeans

    # Check that the fit method works with sample weights
    sample_weight = np.ones(X.shape[0])
    sample_weight[0] = 0.0
    dpmeans.fit(X, sample_weight=sample_weight)

    # Check that the fit method works with sparse data
    X_sparse = X[:500]
    dpmeans.fit(X_sparse)

    # Check that the fit method works with a maximum number of clusters
    dpmeans = DPMeans(n_clusters=1, max_clusters=10, delta=1, random_state=42)
    dpmeans.fit(X)
    assert dpmeans.n_clusters == 10
