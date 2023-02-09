# Revisiting DP-Means: Fast Scalable Algorithms via Parallelism and Delayed Cluster Creation
[Paper](https://openreview.net/pdf?id=rnzVBD8jqlq) <br>

## Introduction
DP-means (Kulis and Jordan, ICML 2012), a nonparametric generalization of K-means, extends the latter to the case where the
number of clusters is unknown. Unlike K-means, however, DP-means is hard to parallelize, a limitation hindering its usage in large-scale tasks. This work bridges this practicality gap by rendering the DP-means approach a viable, fast, and highly-scalable solution. In our paper, we first study the strengths and weaknesses of previous attempts to parallelize the DP-means algorithm. Next, we propose a new parallel algorithm, called PDC-DP-Means (Parallel Delayed Cluster DP-Means), based in part on delayed creation of clusters. Compared with DP-Means, PDC-DP-Means provides not only a major speedup but also performance gains. Finally, we propose two extensions of PDC-DP-Means. The first combines it with an existing method, leading to further speedups. The second extends PDC-DP-Means to a Mini-Batch setting (with an optional support for an online mode), allowing for another major speedup. We verify the utility of the pro-posed methods on multiple datasets. We also show that the proposed methods outperform other non-parametric methods (e.g., DBSCAN). Our highly-efficient code, available in this git repository, can be used to reproduce our experiments. 

## Code


The supplied code has 3 parts -

* The cluster directory, which contains an extension to sklearn with our proposed algorithms, PDC-DP-Means and its MiniBatch version.
* the file `date_pdpmeans.py` which contains our implementation of DACE (in three versions, see below) and PDP-Means.
* Three notebooks that contain the experiment with the other non-parametric methods.

### PDC-DP-Means and MiniBatch PDC-DP-Means

In order to install this, you must clone scikit-learn from: `https://github.com/scikit-learn/scikit-learn.git`.

Navigate to the directory `sklearn/cluster` and replace the files `__init__.py`, `_k_means_lloyd.pyx` and `_kmeans.py` with the respective files under the `cluster` directory.
Next, you need to install sklearn from source. To do so, follow the directions here: https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge.

Now, in order to use it, you can simply use `from sklearn.cluster import MiniBatchDPMeans, DPMeans`. In general, the parameters are the same as the `K-Means` counterpart:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

The only differences are:
1) instead of the `n_clusters` parameter (which stands, in K-Means, for the number fo clusters), there is a new parameter called `delta` (in our papers it was lambda but avoided this vairable name here since lambda is a reserved word in Python);
2) When DPMeans is used the `algorithm` parameter is removed.

### DACE and PDP-Means

In the file `dace_dpmeans.py` there are 4 relevant algorithms -

`parallel_dp(data,delta,processes,iters)' - PDP-Means.  As before, delta replaces lambda, `data' is the data, 'processes' is the amount of parallelization, and `iters' is the maximum iterations (it will stop before if converged).

`DACE(data,delta,num_of_processes)` - The original DACE algorithm. as before, delta replaces lambda, 'data' is the data, num_of_processes is the amount of parallelization.

`DACE_DDP(data,delta,num_of_processes)` - DACE using PDC-DP-Means, but with no inner parallelization.

`DACE_DDP_SPAWN(data,delta,num_of_processes)` - DACE using PDP-DP-Means with inner parallelization, due to different Multi Processing scheme, this might take abit longer to start.


Note that in order to run this file some extra dependencies are required, `evaluations.py` file contain several functions, and while some packages required are quite standard - `torchvision,scikit-learn,annoy,pandas,numpy`, it is also required to have a valid `R` enviroment, and the `R` package `maotai` installed, and the python-R interface package `rpy2`.


### Experiment notebooks
We have included the experiments which does not require additional installations apart from the build-from-source scikit-learn, the three attached notebooks are used to recreate the experiments with the other non-parametric methods. Note that the blackbox optimization (while we supplied the code to run it), need to run separately, as it's multiprocess does not play well with Jupyter Notebook. 


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
While our additional code is under the GPL license, some code snippets here are taken from sklearn, and using them (e.g., the code under the `cluster` directory) is subject to sklearn's license.
