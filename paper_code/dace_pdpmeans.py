from multiprocessing import Process, Queue
from rpy2.robjects import numpy2ri

numpy2ri.activate()
from rpy2.robjects.packages import importr
from time import time

maotai = importr("maotai")
import numpy as np
from sklearn.cluster import KMeans, DPMeans
from sklearn.metrics import pairwise_distances_argmin_min
from evaluations import *
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
import multiprocessing as mp
from annoy import AnnoyIndex
import pandas as pd


def dpmeans_chunk_ddp(q, data, delta, max_iter=1000):
    dpmeans = ddp_nopar(data, delta, max_iter=max_iter)
    q.put(dpmeans)


def dpmeans_chunk_ddp_spawn(q, data, delta):
    dpmeans = (
        DPMeans(init="k-means++", n_clusters=1, n_init=1, delta=delta, max_iter=500)
        .fit(data)
        .cluster_centers_
    )
    q.put(dpmeans)


def dpmeans_chunk(q, data, delta):
    dpres = maotai.dpmeans(data, delta, 1234, 1e-6, True)
    q.put(dpres[1])


def DACE_DDP(data, delta, num_of_processes):
    f = data.shape[1]
    t = AnnoyIndex(f, "euclidean")
    for i in range(data.shape[0]):
        v = data[i].tolist()
        t.add_item(i, v)
    t.build(10)  # 10 trees
    # t.save('test.ann')  # save
    labels = np.ones(data.shape[0]) * -1
    chunk_size = int(data.shape[0] / num_of_processes)
    min_index = 0
    for i in range(num_of_processes):
        if i == num_of_processes - 1:
            chunk_size += data.shape[0] % num_of_processes
        cur_indices = t.get_nns_by_item(
            min_index, data.shape[0], include_distances=False
        )
        cur_size = 0
        for ind in cur_indices:
            if labels[ind] == -1:
                labels[ind] = i
                cur_size += 1
            if cur_size >= chunk_size:
                break
        if i == num_of_processes - 1:
            break
        min_index = np.argwhere(labels == -1)[0][0]

    ps = []
    centers = []
    # ctx = mp.get_context('spawn')
    # q = ctx.Queue()

    q = Queue()
    for i in range(num_of_processes):
        # dpmeans_chunk_ddp(q,data[kmeans.labels_==i],delta)
        # p = ctx.Process(target = dpmeans_chunk_ddp, args = (q,data[kmeans.labels_==i],delta,))
        p = Process(
            target=dpmeans_chunk_ddp,
            args=(
                q,
                data[labels == i],
                delta,
            ),
        )
        p.start()
        ps.append(p)
    for i in range(num_of_processes):
        centers.append(q.get())
        print(f"Got chunk {i+1}/{num_of_processes}")
    all_centers = np.vstack(centers)
    return all_centers


def DACE_DDP_SPAWN(data, delta, num_of_processes):
    f = data.shape[1]
    t = AnnoyIndex(f, "euclidean")
    for i in range(data.shape[0]):
        v = data[i].tolist()
        t.add_item(i, v)
    t.build(10)  # 10 trees
    # t.save('test.ann')  # save
    labels = np.ones(data.shape[0]) * -1
    chunk_size = int(data.shape[0] / num_of_processes)
    min_index = 0
    for i in range(num_of_processes):
        if i == num_of_processes - 1:
            chunk_size += data.shape[0] % num_of_processes
        cur_indices = t.get_nns_by_item(
            min_index, data.shape[0], include_distances=False
        )
        cur_size = 0
        for ind in cur_indices:
            if labels[ind] == -1:
                labels[ind] = i
                cur_size += 1
            if cur_size >= chunk_size:
                break
        if i == num_of_processes - 1:
            break
        min_index = np.argwhere(labels == -1)[0][0]

    ps = []
    centers = []
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    q = Queue()
    for i in range(num_of_processes):
        p = ctx.Process(
            target=dpmeans_chunk_ddp_spawn,
            args=(
                q,
                data[labels == i],
                delta,
            ),
        )
        p.start()
        ps.append(p)
    for i in range(num_of_processes):
        centers.append(q.get())
        print(f"Got chunk {i+1}/{num_of_processes}")
    all_centers = np.vstack(centers)
    return all_centers


def DACE(data, delta, num_of_processes):
    f = data.shape[1]
    t = AnnoyIndex(f, "euclidean")
    for i in range(data.shape[0]):
        v = data[i].tolist()
        t.add_item(i, v)
    t.build(10)  # 10 trees
    # t.save('test.ann')  # save
    labels = np.ones(data.shape[0]) * -1
    chunk_size = int(data.shape[0] / num_of_processes)
    min_index = 0
    for i in range(num_of_processes):
        if i == num_of_processes - 1:
            chunk_size += data.shape[0] % num_of_processes
        cur_indices = t.get_nns_by_item(
            min_index, data.shape[0], include_distances=False
        )
        cur_size = 0
        for ind in cur_indices:
            if labels[ind] == -1:
                labels[ind] = i
                cur_size += 1
            if cur_size >= chunk_size:
                break
        if i == num_of_processes - 1:
            break
        min_index = np.argwhere(labels == -1)[0][0]

    # kmeans = KMeans(init="k-means++", n_clusters=num_of_processes, n_init=1,max_iter=1).fit(data)
    q = Queue()
    ps = []
    centers = []
    for i in range(num_of_processes):
        p = Process(
            target=dpmeans_chunk,
            args=(
                q,
                data[labels == i],
                delta,
            ),
        )
        p.start()
        ps.append(p)
    for i in range(num_of_processes):
        centers.append(q.get())
        print(f"Got chunk {i+1}/{num_of_processes}")
    all_centers = np.vstack(centers)
    return all_centers


def evaluate_dpmeans_model(dpmeans_centers, data, gt, eval_metrics):
    preds = pairwise_distances_argmin(data, dpmeans_centers)
    return {x: metrics_dict[x](preds, gt, data) for x in eval_metrics}


def ddp_nopar(data, delta, max_iter=400):
    centers = np.mean(data, axis=0).reshape(1, -1)
    N = data.shape[0]
    D = data.shape[1]
    labels = np.ones(N) * (-1)
    min_index = -1
    prev_obj = calc_dpmeans_objective(centers, data, delta) / (D * N)
    for j in range(max_iter):
        min_arg, min_dist = pairwise_distances_argmin_min(data, centers)
        max_index = np.argmax(min_dist)
        max_dist = min_dist[max_index]
        labels = min_arg
        if max_dist**2 > delta:
            labels[max_index] = len(centers)
            centers = np.vstack([centers, data[max_index].reshape(1, -1)])
        for c in range(len(centers)):
            if np.sum(labels == c) > 0:
                centers[c, :] = data[labels == c].mean(axis=0)
        new_obj = calc_dpmeans_objective(centers, data, delta) / (D * N)
        delta_obj = np.abs(prev_obj - new_obj)
        prev_obj = new_obj
        print(delta_obj, len(centers))
        if delta_obj < 0.00003:
            break
    return centers


def ddp_nopar_forp(q_centers, q_labels, data, centers, delta, start_index):
    N = data.shape[0]
    D = data.shape[1]

    min_index = -1
    prev_obj = calc_dpmeans_objective(centers, data, delta) / (D * N)
    min_arg, min_dist = pairwise_distances_argmin_min(data, centers)
    q_labels.put(
        (
            start_index,
            min_arg,
        )
    )
    max_index = np.argmax(min_dist)
    max_dist = min_dist[max_index]
    labels = min_arg
    max_label = np.argmin(max_dist)
    center_candidates = []
    new_center_labels = []
    if max_dist**2 > delta:
        new_center_labels.append([max_label + start_index])
        new_centers = data[max_label].reshape(1, -1)
        q_centers.put([(new_center_labels[0], new_centers[0])])
    else:
        q_centers.put([])


def parallel_dp_batch(q_centers, q_labels, data, centers, delta, label_start_index):
    min_arg, min_dist = pairwise_distances_argmin_min(data, centers)
    center_candidates = []
    new_centers = np.zeros((0, data.shape[1]))
    new_center_labels = []
    for i in range(data.shape[0]):
        if min_dist[i] ** 2 > delta:
            if len(new_centers) > 0:
                new_min_arg, new_min_dist = pairwise_distances_argmin_min(
                    data[i].reshape(1, -1), new_centers
                )
                if new_min_dist[0] < delta:
                    new_center_labels[new_min_arg[0]].append(i + label_start_index)
                    continue
            new_centers = np.vstack([new_centers, data[i]])
            new_center_labels.append([i + label_start_index])
            # center_candidates.append((i+label_start_index,data[i],))
    for i in range(len(new_centers)):
        center_candidates.append(
            (
                new_center_labels[i],
                new_centers[i],
            )
        )
    q_labels.put(
        (
            label_start_index,
            min_arg,
        )
    )
    q_centers.put(center_candidates)


def dpvalidate(centers, max_cluster, delta):
    if len(centers) == 0:
        return [], []
    new_centers = centers[0][1].reshape(1, -1)
    new_labels = [max_cluster]
    for i in range(1, len(centers)):
        min_arg, min_dist = pairwise_distances_argmin_min(
            centers[i][1].reshape(1, -1), new_centers
        )
        if min_dist**2 > delta:
            new_labels += [max_cluster + new_centers.shape[0] for x in centers[i][0]]
            new_centers = np.vstack([new_centers, centers[i][1]])
        else:
            new_labels += [max_cluster + min_arg for x in centers[i][0]]
    return new_centers, new_labels


def calc_dpmeans_objective(clusters, data, delta):
    obj = pairwise_distances(data, clusters).min(axis=1).sum() + len(clusters) * delta
    return obj


def parallel_dp(data, delta, processes, iters):
    N = data.shape[0]
    D = data.shape[1]
    chunk_size = int(np.floor(N / processes))
    chunks = []
    for i in range(processes):
        chunk = data[i * chunk_size : chunk_size * (i + 1)]
        chunks.append(chunk)
    centers = np.mean(data, axis=0).reshape(1, -1)
    labels_q = Queue()
    centers_q = Queue()
    labels = np.zeros(N)
    max_cluster = 1
    prev_obj = calc_dpmeans_objective(centers, data, delta) / (D * N)
    print(prev_obj)
    centers_history = [centers]
    iter_time = [0.0]
    for i in range(iters):
        tick = time.time()
        if processes > 1:
            for j in range(processes):
                p = Process(
                    target=parallel_dp_batch,
                    args=(
                        centers_q,
                        labels_q,
                        chunks[j],
                        centers,
                        delta,
                        chunk_size * j,
                    ),
                )
                p.start()
        else:
            parallel_dp_batch(centers_q, labels_q, chunks[0], centers, delta, 0)

        for j in range(processes):
            new_labels = labels_q.get()
            labels[new_labels[0] : new_labels[0] + chunk_size] = new_labels[1]
        new_centers = []
        for j in range(processes):
            new_centers += centers_q.get()
        filtered_centers, new_labels = dpvalidate(new_centers, max_cluster, delta)
        if len(new_labels) > 0:
            # print(
            centers = np.vstack([centers, filtered_centers])
            label_indices = sum(
                [new_centers[i][0] for i in range(len(new_centers))], []
            )
            for i in range(len(new_centers)):
                labels[new_centers[i][0]] = new_labels[i]
        max_cluster = centers.shape[0]
        for c in range(max_cluster):
            if np.sum(labels == c) > 0:
                centers[c, :] = data[labels == c].mean(axis=0)
        cur_obj = calc_dpmeans_objective(centers, data, delta) / (D * N)
        delta_obj = np.abs(prev_obj - cur_obj)
        prev_obj = cur_obj
        print(delta_obj, max_cluster)
        iter_time.append(time.time() - tick)
        centers_history.append(centers)
        if delta_obj < 0.0001:
            break
    return centers, iter_time, centers_history


if __name__ == "__main__":
    ####MNIST:
    EXP_NAME = "MNIST"
    ds = MNIST("data", train=True, download=True)
    D = 16
    K_count = 10

    data = ds.data.numpy()
    gt = ds.targets.numpy()

    data = data.reshape(data.shape[0], -1)

    pca = PCA(n_components=D).fit(data)

    data = pca.transform(data)

    data -= data.mean(axis=0)

    data /= data.std(axis=0)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        data, gt, 0.01, 0.1
    )

    all_times = []
    all_res = []

    for i in range(1):
        tik = time.time()
        # centers = DACE(X_train,24,8)
        # centers = DACE_DDP(X_train,24,8)
        centers = DACE_DDP_SPAWN(X_train, np.sqrt(64), 8)
        all_times.append(time.time() - tik)
        all_res.append(evaluate_dpmeans_model(centers, X_test, y_test, [NMI, K]))
        print(all_times[-1])
        print(all_res[-1])

    print(all_times)
    print(all_res)
    print(np.mean(all_times))
    print(np.std(all_times))
    print(np.mean([x["NMI"] for x in all_res]))
    print(np.std([x["NMI"] for x in all_res]))
