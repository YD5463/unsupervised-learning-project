import json
import os
import random
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from matplotlib import pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, silhouette_score
from tqdm import tqdm
import seaborn as sns
from umap import UMAP

from src.flows_utils.algorithms import dim_reduction_algorithms, clustering_algorithms, anomaly_detection_algorithms, \
    hierarchical_clustering
from src.flows_utils.utils import reduction_algo_wrapper, find_best_algo
from pathlib import Path

from src.vis.visualizations import anomaly_external_var_to_mi, elbow_method

CACHE_PATH = "cache-dynamic-new/"
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)
EXTERNAL_VARS = ["gas_type", "concentration"]
DIMENTIONS_OPTIONS = [2, 10]
NUM_CLUSTERS_OPTIONS = [2, 6, 12, 20]
NUM_OF_CVS = 3
# CV_SIZE = 2600
DATA_PATH = "data/driftdataset"

random.seed(10)
cvs = [[
    random.randint(0, 9),
    random.randint(0, 9),
    random.randint(0, 9),
    random.randint(0, 9)
]
    for i in range(NUM_OF_CVS)
]


def load_dynamic_dataset():
    X_cvs, y_cvs = [], []
    for filename in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, filename), "r") as file:
            df_rows = []
            for line in file.readlines():
                curr_row = {}
                line = line.split(";")
                curr_row["gas_type"] = line[0]
                line = line[1].split(" ")
                curr_row["concentration"] = line[0]

                for sensor_value in line[1:]:
                    sensor_value = sensor_value.split(":")
                    if len(sensor_value) == 2:
                        curr_row[f"sensor_{sensor_value[0]}"] = float(sensor_value[1])
                df_rows.append(curr_row)
            df = pd.DataFrame(df_rows)
            X = df.drop(EXTERNAL_VARS, axis=1).values
            y = df[EXTERNAL_VARS]
            y["concentration"] = y["concentration"].apply(lambda val: int(float(val)))
            X_cvs.append(X)
            y_cvs.append(y)
    return X_cvs, y_cvs


def update_scores(labels: np.ndarray, cv_data, cv_y: pd.DataFrame, lengths, scores: Dict[str, List[float]]):
    sil_score = silhouette_score(cv_data, labels)
    model = CategoricalHMM(n_components=cv_y["gas_type"].nunique()).fit(labels.reshape(-1, 1), lengths=lengths)
    hidden_states_gas_type = model.predict(labels.reshape(-1, 1))
    mi_gas_type_mi = normalized_mutual_info_score(
        cv_y["gas_type"].values,
        hidden_states_gas_type
    )
    model = CategoricalHMM(n_components=cv_y["concentration"].nunique()).fit(labels.reshape(-1, 1), lengths=lengths)
    hidden_states_concentration = model.predict(labels.reshape(-1, 1))
    mi_concentration = normalized_mutual_info_score(
        cv_y["concentration"].values,
        hidden_states_concentration
    )
    scores["mi_concentration_scores"].append(mi_concentration)
    scores["mi_gas_type_scores"].append(mi_gas_type_mi)
    scores["weighted_scores"].append(
        (mi_concentration + mi_gas_type_mi + (sil_score + 1) / 2) / 3
    )
    return scores


def main():
    X_cvs, y_cvs = load_dynamic_dataset()
    best_config_by_clustering = dict()
    for clustering_algo_name in clustering_algorithms.keys():
        dim_reduction_meta: Dict[str, Dict[str, Any]] = dict()
        for reduction_algo_name in dim_reduction_algorithms.keys():
            max_score = float("-inf")
            for dim_num in DIMENTIONS_OPTIONS:
                for k_clusters in NUM_CLUSTERS_OPTIONS:
                    scores = defaultdict(list)
                    for cv_id, in_index in enumerate(cvs):
                        cv_data = np.concatenate([X_cvs[i] for i in range(len(X_cvs)) if i in in_index])
                        cv_y = pd.concat([y_cvs[i] for i in range(len(y_cvs)) if i in in_index])
                        lengths = [X_cvs[i].shape[0] for i in range(len(X_cvs)) if i in in_index]
                        try:
                            cv_data = reduction_algo_wrapper(reduction_algo_name, dim_num, cv_data, cv_id, CACHE_PATH)
                            print(f"doing {clustering_algo_name} for {cv_data.shape} by {reduction_algo_name}")
                            labels = clustering_algorithms[clustering_algo_name](k_clusters, cv_data)
                            scores = update_scores(labels, cv_data, cv_y, lengths, scores)
                        except Exception as e:
                            print(e)
                            break
                    avg_score = np.mean(scores["weighted_scores"])
                    if avg_score > max_score:
                        max_score = avg_score
                        dim_reduction_meta[reduction_algo_name] = {
                            **scores,
                            "avg_score": avg_score,
                            "dim_num": dim_num,
                            "cluster_num": k_clusters,
                            "reduction_algo_name": reduction_algo_name
                        }
        best_algo_name, p_value, t_test_p_value, msg = find_best_algo({
            key: value["weighted_scores"] for key, value in dim_reduction_meta.items()
        })
        best_config_by_clustering[clustering_algo_name] = {
            **dim_reduction_meta[best_algo_name],
            "annova": p_value,
            "t-test": t_test_p_value,
            "msg": msg
        }
        print(f"picking for {clustering_algo_name}: {best_config_by_clustering[clustering_algo_name]}")
    print(best_config_by_clustering)
    best_config_by("weighted_scores", best_config_by_clustering)
    best_config_by("mi_gas_type_scores", best_config_by_clustering)
    best_config_by("mi_concentration_scores", best_config_by_clustering)
    find_best_external_var_per_clustering(X_cvs, y_cvs, best_config_by_clustering)


def find_best_external_var_per_clustering(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame],
                                          best_config_by_clustering: Dict[str, Dict[str, Any]]):
    best_external_var_per_clustering = dict()
    for clustering_algo_name, best_config in best_config_by_clustering.items():
        all_mi = dict()
        print(f"-----------{clustering_algo_name}-------")
        for external_var_name in EXTERNAL_VARS:
            scores = []
            clustering_algo = clustering_algorithms[clustering_algo_name]
            for cv_id, in_index in enumerate(cvs):
                cv_data = np.concatenate([X_cvs[i] for i in range(len(X_cvs)) if i in in_index])
                cv_y = pd.concat([y_cvs[i] for i in range(len(y_cvs)) if i in in_index])
                lengths = [X_cvs[i].shape[0] for i in range(len(X_cvs)) if i in in_index]
                try:
                    cv_data = reduction_algo_wrapper(
                        best_config["reduction_algo_name"],
                        best_config["dim_num"],
                        cv_data, cv_id, CACHE_PATH
                    )
                    labels = clustering_algo(cv_y[external_var_name].nunique(), cv_data)
                    model = CategoricalHMM(
                        n_components=cv_y[external_var_name].nunique()
                    ).fit(
                        labels.reshape(-1, 1),
                        lengths=lengths
                    )
                    hidden_states = model.predict(labels.reshape(-1, 1))
                    scores.append(mutual_info_score(hidden_states, cv_y[external_var_name].values))
                except Exception as e:
                    print(e)
                    scores.append(-1)
            all_mi[external_var_name] = scores
        best_var, p_value, t_test_p_value, msg = find_best_algo(all_mi)
        best_external_var_per_clustering[clustering_algo_name] = {
            "scores": all_mi[best_var],
            "best_var": best_var,
            "anova": p_value,
            "t_test": t_test_p_value,
            "msg": msg
        }
    print(best_external_var_per_clustering)
    with open(f"best_external_var_per_clustering_dynamic.json", "w") as file:
        json.dump(best_external_var_per_clustering, file)
    return best_external_var_per_clustering


def best_config_by(key: str, best_config_by_clustering: Dict):
    clustering_scores = dict()
    for clustering_algo_name, metadata in best_config_by_clustering.items():
        clustering_scores[clustering_algo_name] = metadata[key]
    best_algo_name, p_value, t_test_p_value, msg = find_best_algo(clustering_scores)
    result = {
        "best_algo_name": best_algo_name,
        "annova": p_value,
        "t_test_p_value": t_test_p_value,
        "config": best_config_by_clustering[best_algo_name],
        "msg": msg,
        "all_config": best_config_by_clustering
    }
    print(f"---------final_result_{key}--------")
    print(result)
    with open(f"final_result_{key}.json", "w") as file:
        json.dump(result, file)


def external_var_to_anomalies(X_cvs, y_cvs, external_vars):
    results = list()
    for anomaly_algo_name, anomaly_algo in tqdm(anomaly_detection_algorithms.items()):
        if anomaly_algo is None:
            continue
        scores = defaultdict(list)
        for cv_id, in_index in enumerate(cvs):
            cv_data = np.concatenate([X_cvs[i] for i in range(len(X_cvs)) if i in in_index])
            cv_y = pd.concat([y_cvs[i] for i in range(len(y_cvs)) if i in in_index])
            lengths = [X_cvs[i].shape[0] for i in range(len(X_cvs)) if i in in_index]
            labels = anomaly_algo.fit_predict(cv_data)
            labels[labels == -1] = 0
            for external_var_name in external_vars:
                model = CategoricalHMM(n_components=cv_y[external_var_name].nunique()).fit(
                    labels.reshape(-1, 1),
                    lengths=lengths
                )
                labels = model.predict(labels.reshape(-1, 1))
                scores[external_var_name].append(
                    mutual_info_score(
                        labels,
                        cv_y[external_var_name]
                    )
                )

        for external_var_name, external_var_scores in scores.items():
            results.append({
                "algo_name": anomaly_algo_name,
                "external_var": external_var_name,
                "MI": np.mean(external_var_scores)
            })

    return pd.DataFrame(results)


def plot_mi_to_anomaly():
    # X_cvs, y_cvs = load_dynamic_dataset()
    # df = external_var_to_anomalies(X_cvs, y_cvs, EXTERNAL_VARS)
    # df.to_csv("save.csv")
    df = pd.read_csv("save.csv")
    anomaly_external_var_to_mi(df)
    print(df)


def plot_stuff():
    X_cvs, y_cvs = load_dynamic_dataset()
    in_index = cvs[0]
    X = np.concatenate([X_cvs[i] for i in range(len(X_cvs)) if i in in_index])
    y = pd.concat([y_cvs[i] for i in range(len(y_cvs)) if i in in_index])
    lengths = [X_cvs[i].shape[0] for i in range(len(X_cvs)) if i in in_index]
    external_var = "concentration"
    n_clusters = y[external_var].nunique()
    labels = hierarchical_clustering(
        n_clusters,
        MDS(n_components=10, n_jobs=-1).fit_transform(X)
    )
    model = CategoricalHMM(
        n_components=y[external_var].nunique()
    ).fit(
        labels.reshape(-1, 1),
        lengths=lengths
    )
    labels = model.predict(labels.reshape(-1, 1))
    y_true = y[external_var].values
    # y_true = most_similar_permutation(y_true, labels)
    elbow_method(X, labels, n_clusters)
    # X = UMAP(
    #     n_neighbors=100, n_components=2,
    #     n_epochs=1000, init='spectral',
    #     low_memory=False, verbose=False
    # ).fit_transform(X)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # ax1.scatter(X[:, 0], X[:, 1], c=labels)
    # ax2.scatter(X[:, 0], X[:, 1], c=y_true)
    # plt.savefig("static_clustering_hierarchical_clustering_genre_top.png")
    # plt.show()


if __name__ == '__main__':
    plot_stuff()
    # main2()
    # main3()

