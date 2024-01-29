import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from pdc_dp_means import MiniBatchDPMeans
from sklearn.metrics import normalized_mutual_info_score

def cdist(XA, XB):
    return np.sqrt(((XA[:, None] - XB)**2).sum(-1))


def create_stream_from_data(data, gt, sample_count):
    parts = int(np.floor(len(gt) / sample_count))
    stream_batches = []
    stream_labels = []
    for i in range(parts):
        stream_batches.append(data[i * sample_count : (i + 1) * sample_count, :])
        stream_labels.append(gt[i * sample_count : (i + 1) * sample_count])
    return stream_batches, stream_labels


def test_minibatch_dpmeans_partial_fit():
    # Generate some random data
    X, y = make_blobs(n_samples=3000, centers=5, random_state=42)

    # Create a MiniBatchDPMeans object
    dpmeans = MiniBatchDPMeans(n_clusters=1, delta=20, random_state=42, batch_size=100)

    # Create a stream of data from the dataset
    stream_batches, stream_labels = create_stream_from_data(X, y, sample_count=100)

    # Partially fit the MiniBatchDPMeans object to the stream of data
    for i in range(len(stream_batches)):
        dpmeans.partial_fit(stream_batches[i])

    # Check that the number of clusters is correct
    assert dpmeans.n_clusters == 5

    # Check that the cluster centers are close to the true centers
    true_centers = np.array([X[y == i].mean(axis=0) for i in range(5)])
    # Compute the pairwise distances between the true centers and the estimated centers
    distances = cdist(true_centers, dpmeans.cluster_centers_)

    # Find the minimum distance between each true center and any estimated center
    min_distances = np.min(distances, axis=1)

    # Assert that the minimum distance is less than or equal to the tolerance
    assert np.all(min_distances <= 2.0)

    # Compute the NMI between the true labels and the labels computed by MiniBatchDPMeans
    mbdp_means_labels = pairwise_distances_argmin(X, dpmeans.cluster_centers_)
    nmi = normalized_mutual_info_score(y, mbdp_means_labels)
    # Assert that the NMI is greater than or equal to a certain threshold (e.g., 0.9)
    assert nmi >= 0.85

    # Check that the partial_fit method returns self
    assert dpmeans.partial_fit(X) is dpmeans

    # Check that the partial_fit method works with sample weights
    sample_weight = np.ones(X.shape[0])
    sample_weight[0] = 0.0
    dpmeans.partial_fit(X, sample_weight=sample_weight)


def test_minibatch_dpmeans():
    # Generate some random data
    X, y = make_blobs(n_samples=3000, centers=10, random_state=42)

    # Create a MiniBatchDPMeans object and fit the data
    dpmeans = MiniBatchDPMeans(n_clusters=1, delta=20, random_state=42)
    dpmeans.fit_predict(X)

    # Check that the number of clusters is correct
    assert dpmeans.n_clusters == 10

    # Check that the cluster centers are close to the true centers
    true_centers = np.array([X[y == i].mean(axis=0) for i in range(6)])
    # Compute the pairwise distances between the true centers and the estimated centers
    distances = cdist(true_centers, dpmeans.cluster_centers_)

    # Find the minimum distance between each true center and any estimated center
    min_distances = np.min(distances, axis=1)

    # Assert that the minimum distance is less than or equal to the tolerance
    print(min_distances)
    assert np.all(min_distances <= 2.0)

    # Compute the NMI between the true labels and the labels computed by MiniBatchDPMeans
    nmi = normalized_mutual_info_score(y, dpmeans.labels_)
    # Assert that the NMI is greater than or equal to a certain threshold (e.g., 0.9)
    assert nmi >= 0.85

    # Check that the fit method returns self
    assert dpmeans.fit(X) is dpmeans

    # Check that the fit method works with sample weights
    sample_weight = np.ones(X.shape[0])
    sample_weight[0] = 0.0
    dpmeans.fit(X, sample_weight=sample_weight)


if __name__ == "__main__":
    test_minibatch_dpmeans_partial_fit()
    test_minibatch_dpmeans()