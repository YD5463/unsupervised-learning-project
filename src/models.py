import json
import random
from collections import defaultdict
from typing import Tuple, Dict, Any, List

import numpy as np
import os
import pandas as pd
from src.algorithms import clustering_algorithms, dim_reduction_algorithms, anomaly_detection_algorithms
from sklearn.metrics import mutual_info_score
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
from tqdm import tqdm
from pathlib import Path

OUTPUT_PATH = "output/"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# Config
random_state = 0
data_path = "../data/fma_metadata"
features = pd.read_csv(os.path.join(data_path, "features.csv"), index_col=0, header=[0, 1, 2])
tracks = pd.read_csv(os.path.join(data_path, "tracks.csv"), index_col=0, header=[0, 1])
dimentions_options = [10, None]
num_clusters_options = list(range(2, 20, 10))
num_of_cvs = 5
cv_size = 100
p_value_thr = 0.05
external_vars = [('track', 'genre_top'), ('track', 'license'), ('album', 'type')]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    for column in external_vars:
        tracks[column] = tracks[column].astype('category')
    X, y = features, tracks[external_vars]  # X, y
    for column in y:
        print(column, y[column].nunique())
    return X, y


def generate_cvs(X: np.ndarray, y: pd.DataFrame):
    X_cvs = []
    y_cvs = []
    for i in range(num_of_cvs):
        rows = np.random.randint(cv_size, size=X.shape[0]).astype('bool')
        X_cvs.append(X[rows])
        y_cvs.append(y[rows])
    return X_cvs, y_cvs


def main_flow(X_cvs: List[np.ndarray]) -> Dict[str, Dict[str, Any]]:
    best_config_by_clustering = dict()
    for clustering_algo_name, clustering_algo in tqdm(clustering_algorithms.items()):
        dim_reduction_scores = dict()
        dim_reduction_meta = dict()
        for reduction_algo_name, reduction_algo in dim_reduction_algorithms.items():
            print(f"in reduction_algo: {reduction_algo_name}")
            max_score = float("-inf")
            for dim_num in dimentions_options:
                for k_clusters in num_clusters_options:
                    scores = []
                    for cv_data in X_cvs:
                        if reduction_algo:
                            cv_data = reduction_algo(dim_num).fit_transform(cv_data)
                        labels = clustering_algo(k_clusters, cv_data)
                        scores.append(silhouette_score(cv_data, labels))
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

        _, p_value = f_oneway(list(dim_reduction_scores.values()))
        best_config_by_clustering[clustering_algo_name] = random.choice(list(dim_reduction_meta.values()))
        if p_value < p_value_thr:
            sorted_reduction_algos = sorted(
                dim_reduction_meta,
                key=lambda key: dim_reduction_meta[key]["max_score"],
                reverse=True
            )
            reduction_algo_name_1, reduction_algo_name_2 = sorted_reduction_algos[:2]
            _, t_test_p_value = ttest_rel(
                dim_reduction_scores[reduction_algo_name_1],
                dim_reduction_scores[reduction_algo_name_2]
            )
            best_algo_name: str = sorted_reduction_algos[0]
            best_config_by_clustering[clustering_algo_name] = dim_reduction_meta[best_algo_name]
            if t_test_p_value >= p_value_thr:
                print(f"followed by t-test: algorithms {reduction_algo_name_1}, {reduction_algo_name_2} are the same")
        else:
            print(f"followed by annova: algorithms {dim_reduction_scores.keys()} are the same")
        print(f"picking for {clustering_algo_name}: {best_config_by_clustering[clustering_algo_name]}")
        print(best_config_by_clustering)
        with open("output/best_config_by_clustering", "w") as file:
            json.dump(best_config_by_clustering, file)
    return best_config_by_clustering


def second_flow(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame], best_config_by_clustering: Dict[str, Dict[str, Any]]):
    # hidden variables with the best clustering and dim_reduction
    best_cluster_algo_per_external_var = dict()
    for external_var_name in external_vars:
        all_mi = dict()
        for clustering_algo, best_config in best_config_by_clustering.items():
            scores = []
            for cv_data, cv_y_true in zip(X_cvs, y_cvs):
                reduction_algo = dim_reduction_algorithms[best_config["reduction_algo_name"]]
                if reduction_algo:
                    cv_data = reduction_algo(best_config["max_dim_num"]).fit_transform(cv_data)
                labels = clustering_algorithms[clustering_algo](best_config["max_cluster_num"], cv_data)
                scores.append(mutual_info_score(labels, cv_y_true[external_var_name].values))
            all_mi[clustering_algo] = scores

        _, p_value = f_oneway(all_mi.values())
        best_cluster_algo_per_external_var[external_var_name] = random.choice(list(all_mi.values()))
        if p_value < p_value_thr:
            sorted_by_algo_scores = sorted(
                all_mi,
                key=lambda key: np.mean(all_mi[key]),
                reverse=True
            )
            algo_name_1, algo_name_2 = sorted_by_algo_scores[:2]
            _, t_test_p_value = ttest_rel(all_mi[algo_name_1], all_mi[algo_name_2])
            best_algo_name: str = sorted_by_algo_scores[0]
            best_cluster_algo_per_external_var[external_var_name] = {
                "algo_name": best_algo_name,
                "scores": all_mi[best_algo_name]
            }
            if t_test_p_value >= p_value_thr:
                print(f"followed by t-test: algorithms {algo_name_1}, {algo_name_2} are the same")
        else:
            print(f"followed by annova: algorithms {all_mi.keys()} are the same")
    print(best_cluster_algo_per_external_var)
    with open("output/best_cluster_algo_per_external_var.json", "w") as file:
        json.dump(best_cluster_algo_per_external_var, file)
    return best_cluster_algo_per_external_var


def third_flow(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame], best_config_by_clustering: Dict[str, Dict[str, Any]]):
    all_mi = dict()
    for clustering_algo, best_config in best_config_by_clustering.items():
        for external_var in external_vars:
            for cv in X_cvs:
                # change in best_config max_cluster_num to len(np.unique(external_var))
                labels = clustering_algorithms[clustering_algo]()
                all_mi[clustering_algo, external_var].append(mutual_info_score(labels, external_var))
        _, p_value = f_oneway(all_mi[clustering_algo, :].values())

        # t-test with the top 2
        # yield best_cluster_algo_per_external_var


def main():
    X, y = load_data()
    rows = np.random.randint(cv_size, size=X.shape[0]).astype('bool')
    X, y = X[rows].values, y[rows]
    X_cvs, y_cvs = generate_cvs(X, y)
    best_config_by_clustering = main_flow(X_cvs)
    second_flow(X, y, best_config_by_clustering)


if __name__ == '__main__':
    main()
