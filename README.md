# Parallel Delayed Cluster DP-Means

[Paper](https://openreview.net/pdf?id=rnzVBD8jqlq) <br>

### Introduction
The PDC-DP-Means package presents a highly optimized version of the DP-Means algorithm, introducing a new parallel algorithm, Parallel Delayed Cluster DP-Means (PDC-DP-Means), and a MiniBatch implementation for enhanced speed. These features cater to scalable and efficient cluster analysis where the number of clusters is unknown.

In addition to offering major speed improvements, the PDC-DP-Means algorithm supports an optional online mode for real-time data processing. Its scikit-learn-like interface is user-friendly and designed for easy integration into existing data workflows. PDC-DP-Means outperforms other nonparametric methods, establishing its efficiency and scalability in the realm of clustering algorithms.

See the paper for more details.


### Installation
`pip install pdc-dp-means`

### Quick Start

    from sklearn.datasets import make_blobs
    from pdc_dp_means import DPMeans

    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Apply DPMeans clustering
    dpmeans = DPMeans(n_clusters=1,n_init=10, delta=10)  # n_init and delta parameters
    dpmeans.fit(X)

    # Predict the cluster for each data point
    y_dpmeans = dpmeans.predict(X)

    # Plotting clusters and centroids
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=y_dpmeans, s=50, cmap='viridis')
    centers = dpmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

One thing to note is that we replace the `\lambda` parameter from the paper with `delta` in the code, as `lambda` is a reserved word in python.

### Usage
Please refer to the documentation: https://pdc-dp-means.readthedocs.io/en/latest/

### Paper Code
Please refer to https://github.com/BGU-CS-VIL/pdc-dp-means/tree/main/paper_code for the code used in the paper.

### Citing this work
If you use this code for your work, please cite the following:

```
@inproceedings{dinari2022revisiting,
  title={Revisiting DP-Means: Fast Scalable Algorithms via Parallelism and Delayed Cluster Creation},
  author={Dinari, Or and Freifeld, Oren},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}
```
### License 
Our code is licensed under the BDS-3-Clause license.
