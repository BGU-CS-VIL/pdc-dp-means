# Parallel Delayed Cluster DP-Means

## Revisiting DP-Means: Fast Scalable Algorithms via Parallelism and Delayed Cluster Creation
[Paper](https://openreview.net/pdf?id=rnzVBD8jqlq) <br>

### Introduction
DP-means (Kulis and Jordan, ICML 2012), a nonparametric generalization of K-means, extends the latter to the case where the
number of clusters is unknown. Unlike K-means, however, DP-means is hard to parallelize, a limitation hindering its usage in large-scale tasks. This work bridges this practicality gap by rendering the DP-means approach a viable, fast, and highly-scalable solution. In our paper, we first study the strengths and weaknesses of previous attempts to parallelize the DP-means algorithm. Next, we propose a new parallel algorithm, called PDC-DP-Means (Parallel Delayed Cluster DP-Means), based in part on delayed creation of clusters. Compared with DP-Means, PDC-DP-Means provides not only a major speedup but also performance gains. Finally, we propose two extensions of PDC-DP-Means. The first combines it with an existing method, leading to further speedups. The second extends PDC-DP-Means to a Mini-Batch setting (with an optional support for an online mode), allowing for another major speedup. We verify the utility of the pro-posed methods on multiple datasets. We also show that the proposed methods outperform other non-parametric methods (e.g., DBSCAN). Our highly-efficient code, available in this git repository, can be used to reproduce our experiments. 


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
