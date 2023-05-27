from sklearn.cluster import MiniBatchDPMeans, DPMeans, KMeans, MiniBatchKMeans
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import pair_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import time
import pandas as pd
import os


NMI = "NMI"
PURITY = "Purity"
ARI = "ARI"
SIL = "Sil"
FMEASURE = "Fmeasure"
PRECISION = "precision"
RECALL = "recall"
K = "K"
TIME = "Time"


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def recall(y_true, y_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(y_true, y_pred)
    return tp / (tp + fn)


def precision(y_true, y_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(y_true, y_pred)
    return tp / (tp + fp)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r)


metrics_dict = {
    NMI: lambda pred, gt, data: nmi(gt, pred),
    PURITY: lambda pred, gt, data: purity_score(gt, pred),
    ARI: lambda pred, gt, data: ari(gt, pred),
    SIL: lambda pred, gt, data: sil(data, pred),
    FMEASURE: lambda pred, gt, data: f1(gt, pred),
    PRECISION: lambda pred, gt, data: precision(gt, pred),
    RECALL: lambda pred, gt, data: recall(gt, pred),
    K: lambda pred, gt, data: len(np.unique(pred)),
}


def evaluate_model(model, data, gt, eval_metrics):
    preds = pairwise_distances_argmin(data, model.cluster_centers_)
    return {x: metrics_dict[x](preds, gt, data) for x in eval_metrics}


def run_model(model, data, eval_metrics, test_data, test_gt):
    tic = time.time()
    m = model.fit(data)
    toc = time.time() - tic
    results = evaluate_model(m, test_data, test_gt, eval_metrics)
    results[TIME] = toc
    return results


def run_all_models(models_dict, data, test_data, test_gt, count, eval_metrics):
    results = {}
    for m, v in models_dict.items():
        print(f"Running {m}")
        results[m] = {met: [] for met in (eval_metrics + [TIME])}
        for i in range(count):
            print(f"iter {i}/{count}")
            res = run_model(v(), data, eval_metrics, test_data, test_gt)
            for k, r in res.items():
                results[m][k].append(r)
    mean_stds = {}
    for m, v in models_dict.items():
        mean_stds[m] = {met: [] for met in (eval_metrics + [TIME])}
        for k in mean_stds[m].keys():
            mean_stds[m][k] = (np.mean(results[m][k]), np.std(results[m][k]))
    return results, mean_stds


def save_data(data, gt, exp_name):
    save_path = os.path.join("data", exp_name + ".npy")
    saved_results = {"data": data, "gt": gt}
    np.save(save_path, saved_results)


def load_data(exp_name):
    save_path = save_path = os.path.join("data", exp_name + ".npy")
    saved_results = np.load(save_path, allow_pickle=True)
    # return saved_results
    return saved_results.item()["data"], saved_results.item()["gt"]


def save_results(results, mean_stds, exp_name):
    save_path = os.path.join("results", exp_name + ".npy")
    saved_results = {"full": results, "mean_std": mean_stds}
    np.save(save_path, saved_results)


def load_results(exp_name):
    save_path = os.path.join("results", exp_name + ".npy")
    saved_results = np.load(save_path, allow_pickle=True)
    # return saved_results
    return saved_results.item()["full"], saved_results.item()["mean_std"]


def train_val_test_split(data, labels, val_frac, test_frac):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_frac, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=1
    )  # 0.25 x 0.8 = 0.2
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def calc_objective(model, obj, data, gt):
    return -metrics_dict[obj](
        pairwise_distances_argmin(data, model.cluster_centers_), gt, data
    )
