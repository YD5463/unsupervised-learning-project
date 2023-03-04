import json
import pickle
from typing import Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from umap import UMAP

from src.flows_utils.algorithms import clustering_algorithms, dim_reduction_algorithms, anomaly_detection_algorithms, \
    hierarchical_clustering
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import warnings
from src.flows_utils.utils import get_silhouette_scores, reduction_algo_wrapper, generate_cvs, find_best_algo
from pathlib import Path

from src.vis.visualizations import elbow_method

OUTPUT_PATH = "../output/"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
CACHE_PATH = "../cache/"
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

# Config
# random_state = 0
DATA_PATH = "../data/fma_metadata"
DIMENTIONS_OPTIONS = [10, 50, 100]
NUM_CLUSTERS_OPTIONS = [2, 4, 8, 12, 16, 20]
NUM_OF_CVS = 5
CV_SIZE = 2000
P_VALUE_THR = 0.05
EXTERNAL_VARS = [('track', 'genre_top'), ('track', 'language_code'), ('album', 'type')]


def load_data() -> Tuple[np.ndarray, pd.DataFrame]:
    features = pd.read_csv(os.path.join(DATA_PATH, "features.csv"), index_col=0, header=[0, 1, 2])
    tracks = pd.read_csv(os.path.join(DATA_PATH, "tracks.csv"), index_col=0, header=[0, 1])
    features.columns = features.columns.map('_'.join)
    df = pd.concat([features, tracks[EXTERNAL_VARS]], axis=1)
    df = df.dropna()
    X, y = df.drop(EXTERNAL_VARS, axis=1), df[EXTERNAL_VARS]
    for column in EXTERNAL_VARS:
        y[column] = y[column].astype('category').cat.codes
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    return X, y


def find_best_config_by_clustering(X_cvs: List[np.ndarray]) -> Dict[str, Dict[str, Any]]:
    best_config_by_clustering = dict()
    for clustering_algo_name in clustering_algorithms.keys():
        dim_reduction_scores = dict()
        dim_reduction_meta: Dict[str, Dict[str, Any]] = dict()
        for reduction_algo_name in dim_reduction_algorithms.keys():
            max_score = float("-inf")
            for dim_num in DIMENTIONS_OPTIONS:
                for k_clusters in NUM_CLUSTERS_OPTIONS:
                    scores = get_silhouette_scores(
                        X_cvs, clustering_algo_name,
                        reduction_algo_name, dim_num, k_clusters
                    )
                    curr_score = np.mean(scores)
                    if curr_score > max_score:
                        max_score = curr_score
                        dim_reduction_scores[reduction_algo_name] = scores
                        dim_reduction_meta[reduction_algo_name] = {
                            "max_score": max_score,
                            "max_dim_num": dim_num,
                            "max_cluster_num": k_clusters,
                            "scores": scores,
                            "reduction_algo_name": reduction_algo_name
                        }
        best_algo_name, p_value, t_test_p_value, msg = find_best_algo(dim_reduction_scores)
        best_config_by_clustering[clustering_algo_name] = dim_reduction_meta[best_algo_name]
        print(f"picking for {clustering_algo_name}: {best_config_by_clustering[clustering_algo_name]}")
    return best_config_by_clustering


def find_best_cluster_algo_per_external_var(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame],
                                            best_config_by_clustering: Dict[str, Dict[str, Any]]):
    # hidden variables with the best clustering and dim_reduction
    best_cluster_algo_per_external_var = dict()
    for external_var_name in EXTERNAL_VARS:
        print(f"-----------{external_var_name}-------")
        all_mi = dict()
        for clustering_algo_name, best_config in best_config_by_clustering.items():
            scores = []
            clustering_algo = clustering_algorithms[clustering_algo_name]
            for cv_id, (cv_data, cv_y_true) in enumerate(zip(X_cvs, y_cvs)):
                try:
                    cv_data = reduction_algo_wrapper(
                        best_config["reduction_algo_name"],
                        best_config["max_dim_num"],
                        cv_data, cv_id, CACHE_PATH
                    )
                    labels = clustering_algo(best_config["max_cluster_num"], cv_data)
                    scores.append(mutual_info_score(labels, cv_y_true[external_var_name].values))
                except Exception as e:
                    print(e)
                    scores.append(-1)
            all_mi[clustering_algo_name] = scores
        best_algo_name, p_value, t_test_p_value, msg = find_best_algo(all_mi)
        best_cluster_algo_per_external_var[external_var_name] = {
            "algo_name": best_algo_name,
            "scores": all_mi[best_algo_name]
        }
    print(best_cluster_algo_per_external_var)
    with open(f"{OUTPUT_PATH}/best_cluster_algo_per_external_var.pkl", "wb") as file:
        pickle.dump(best_cluster_algo_per_external_var, file)
    return best_cluster_algo_per_external_var


def find_best_external_var_per_clustering(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame],
                                          best_config_by_clustering: Dict[str, Dict[str, Any]]):
    best_external_var_per_clustering = dict()
    for clustering_algo_name, best_config in best_config_by_clustering.items():
        all_mi = dict()
        print(f"-----------{clustering_algo_name}-------")
        for external_var_name in EXTERNAL_VARS:
            scores = []
            clustering_algo = clustering_algorithms[clustering_algo_name]
            for cv_id, (cv_data, cv_y_true) in enumerate(zip(X_cvs, y_cvs)):
                try:
                    cv_data = reduction_algo_wrapper(
                        best_config["reduction_algo_name"],
                        best_config["max_dim_num"],
                        cv_data, cv_id, CACHE_PATH
                    )
                    labels = clustering_algo(cv_y_true[external_var_name].nunique(), cv_data)
                    scores.append(mutual_info_score(labels, cv_y_true[external_var_name].values))
                except Exception as e:
                    print(e)
                    scores.append(-1)
            all_mi[external_var_name] = scores
        best_var, p_value, t_test_p_value, msg = find_best_algo(all_mi)
        best_external_var_per_clustering[clustering_algo_name] = {
            "scores": all_mi[best_var],
            "best_var": best_var,
            "anova": p_value,
            "t_test": t_test_p_value
        }
    print(best_external_var_per_clustering)
    with open(f"{OUTPUT_PATH}/best_external_var_per_clustering.pkl", "wb") as file:
        pickle.dump(best_external_var_per_clustering, file)
    return best_external_var_per_clustering


def second_part(X_cvs, y_cvs, best_config_by_clustering):
    silhouette_per_clustering = {}
    for clustering_algo_name, best_config in best_config_by_clustering.items():
        scores = get_silhouette_scores(
            X_cvs, clustering_algo_name,
            best_config["reduction_algo_name"],
            best_config["max_dim_num"], best_config["max_cluster_num"]
        )
        silhouette_per_clustering[clustering_algo_name] = scores
    print("\n--------silhouette_per_clustering------------")
    print(silhouette_per_clustering)
    best_external_var_per_clustering = find_best_external_var_per_clustering(
        X_cvs, y_cvs,
        best_config_by_clustering
    )
    print("\n--------best_external_var_per_clustering------------")
    print(best_external_var_per_clustering)
    with open("silhouette_per_clustering.json", "w") as file:
        json.dump(silhouette_per_clustering, file)
    with open("best_external_var_per_clustering.json", "w") as file:
        json.dump(best_external_var_per_clustering, file)


def check_with_anomaly_detection(best_config_by_clustering):
    X, y = load_data()
    # for anomaly_algo_name, anomaly_algo in tqdm(anomaly_detection_algorithms.items()):
    #     if anomaly_algo is None:
    #         continue
    labels = anomaly_detection_algorithms["IsolationForest"].fit_predict(X)
    X_cvs, y_cvs = generate_cvs(X, y, NUM_OF_CVS, CV_SIZE)
    best_cluster_algo_per_external_var = find_best_cluster_algo_per_external_var(
        X_cvs, y_cvs,
        best_config_by_clustering
    )
    print("\n--------best_cluster_algo_per_external_var------------")
    print(best_cluster_algo_per_external_var)


def full_flow():
    X, y = load_data()
    X_cvs, y_cvs = generate_cvs(X, y, NUM_OF_CVS, CV_SIZE)
    # best_config_by_clustering = find_best_config_by_clustering(X_cvs)
    # print("--------best_config_by_clustering------------")
    # print(best_config_by_clustering)
    # best_algo_name, p_value, t_test_p_value = find_best_algo(
    #     {key: value["scores"] for key, value in best_config_by_clustering.items()}
    # )
    with open("../reports/best_config_by_clustering.json", "r") as file:
        best_config_by_clustering: Dict = json.load(file)
    # check_with_anomaly_detection(best_config_by_clustering)
    second_part(X_cvs, y_cvs, best_config_by_clustering)


def plot_stuff():
    X, y = load_data()
    external_var = ('track', 'genre_top')
    n_clusters = y[external_var].nunique()
    # elbow_method(X, labels, n_clusters)
    rows = np.random.randint(X.shape[0], size=1000)
    X = X[rows, :]
    y = y.iloc[rows]
    X = LocallyLinearEmbedding(n_components=10, n_jobs=-1).fit_transform(X)
    X = UMAP(n_neighbors=100, n_components=2, n_epochs=1000, init='spectral', low_memory=False, verbose=False).fit_transform(X)
    labels = hierarchical_clustering(
        n_clusters,
        X
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.scatter(X[:, 0], X[:, 1], c=labels)
    ax2.scatter(X[:, 0], X[:, 1], c=y[external_var].values)
    plt.savefig("static_clustering_hierarchical_clustering_genre_top.png")
    plt.show()


if __name__ == '__main__':
    # full_flow()
    plot_stuff()

