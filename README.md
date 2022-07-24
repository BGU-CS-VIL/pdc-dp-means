## Revisiting DP-Means: Fast Scalable Algorithms via Parallelism and Delayed Cluster Creation
[Paper](https://openreview.net/pdf?id=rnzVBD8jqlq) <br>
The supplied code has 3 parts -

* The cluster directory, which contains an extension to sklearn with our proposed algorithms, PDC-DP-Means and it's MiniBatch version.
* the file `date_pdpmeans.py` which contains our implementation of DACE (in three versions, see below) and PDP-Means.
* Three notebooks which contains the experiment with the other non-parametric methods.

### PDC-DP-Means and MiniBatch PDC-DP-Means

In order to install this, you must clone scikit-learn from: `https://github.com/scikit-learn/scikit-learn.git`.

Navigate the directory `sklearn/cluster`, and replace the files `__init__.py`, `_k_means_lloyd.pyx` and `_kmeans.py` with the respective files under the `cluster` directory.
Next, you need to install sklearn from source, follow the directions here: https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge.

Now, in order to use it, you can simply use `from sklearn.cluster import MiniBatchDPMeans, DPMeans`, the parameters are the same as the `K-Means` counterpart:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

The differences are that instead of number of `n_clusters`, parameter, there is a new parameter `delta` (replacement for lambda, which is reserved in Python),  and in the case of DPMeans, the `algorithm` parameter is removed.

### DACE and PDP-Means

In the file `dace_dpmeans.py` there are 4 relevant algorithms -

`parallel_dp(data,delta,processes,iters)` - PDP-Means , as before, delta replaces lambda, the data is the data, processes is the amount of parallelization, and iters is the maximum iterations (it will stop before if converged).

`DACE(data,delta,num_of_processes)` - The original DACE, as before, delta replaces lambda, the data is the data, num_of_processes is the amount of parallelization.

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
