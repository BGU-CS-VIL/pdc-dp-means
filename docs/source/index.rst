Welcome to PDC-DP-Means documentation!
======================================

**PDC-DP-Means** is a Python library for running fast, scalable DP-Means or Mini-Batch DP-Means. It is built on top scikit-learn and numpy.

Check out the :doc:`usage` section for further information, including
how to :ref:`install <installation>` the project.

Quickstart
----------
.. code-block:: python

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

Contents
--------

.. toctree::

   usage
   DPMeans
   MiniBatchDPMeans




If you use this package for your reseach, please cite the following paper:

.. code-block:: tex

   @inproceedings{dinari2022revisiting,
   title={Revisiting DP-Means: Fast Scalable Algorithms via Parallelism and Delayed Cluster Creation},
   author={Dinari, Or and Freifeld, Oren},
   booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
   year={2022}
   }
