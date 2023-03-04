import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

from src.flows_utils.algorithms import dim_reduction_algorithms, clustering_algorithms
from src.flows_utils.utils import reduction_algo_wrapper, find_best_algo, generate_cvs
from src.static import dimentions_options, num_clusters_options
from pathlib import Path


CACHE_PATH = "cache-dynamic/"
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)
EXTERNAL_VARS = ["gas_type", "concentration"]
DIMENTIONS_OPTIONS = [10, 50, 100]
NUM_CLUSTERS_OPTIONS = [2, 4, 8, 12, 16, 20]
NUM_OF_CVS = 5
CV_SIZE = 2600
DATA_PATH = "data/driftdataset"



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

    cvs = [[0, 1, 2, 3, 6], [7, 8], [4], [5], [9]]
    lengths = [[X_cvs[i].shape[0] for i in cv] for cv in cvs]
    X_cvs = [np.concatenate([X_cvs[i] for i in cv]) for cv in cvs]
    y_cvs = [pd.concat([y_cvs[i] for i in cv]) for cv in cvs]
    return X_cvs, y_cvs, lengths


def update_scores(labels: np.ndarray, cv_y: pd.DataFrame, scores: Dict[str, List[float]], lengths):
    model = CategoricalHMM(n_components=cv_y["gas_type"].nunique()).fit(labels.reshape(-1, 1), lengths)
    hidden_states_gas_type = model.predict(labels.reshape(-1, 1))
    mi_gas_type_mi = normalized_mutual_info_score(
        cv_y["gas_type"].values,
        hidden_states_gas_type
    )
    model = CategoricalHMM(n_components=cv_y["concentration"].nunique()).fit(labels.reshape(-1, 1), lengths)
    hidden_states_concentration = model.predict(labels.reshape(-1, 1))
    mi_concentration = normalized_mutual_info_score(
        cv_y["gas_type"].values,
        hidden_states_concentration
    )
    scores["mi_concentration_scores"].append(mi_concentration)
    scores["mi_gas_type_scores"].append(mi_gas_type_mi)
    scores["weighted_scores"].append(
        (mi_concentration + mi_gas_type_mi) / 2
    )
    return scores


def main():
    X_cvs, y_cvs, lengths = load_dynamic_dataset()
    best_config_by_clustering = dict()
    for clustering_algo_name in clustering_algorithms.keys():
        dim_reduction_meta: Dict[str, Dict[str, Any]] = dict()
        for reduction_algo_name in dim_reduction_algorithms.keys():
            max_score = float("-inf")
            for dim_num in dimentions_options:
                for k_clusters in num_clusters_options:
                    scores = defaultdict(list)
                    for cv_id, (cv_data, cv_y, length) in enumerate(zip(X_cvs, y_cvs, lengths)):
                        try:
                            cv_data = reduction_algo_wrapper(reduction_algo_name, dim_num, cv_data, cv_id, CACHE_PATH)
                            print(f"doing {clustering_algo_name} for {cv_data.shape} by {reduction_algo_name}")
                            labels = clustering_algorithms[clustering_algo_name](k_clusters, cv_data)
                            scores = update_scores(labels, cv_y, scores, length)
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
        best_algo_name, p_value, t_test_p_value = find_best_algo({
            key: value["weighted_scores"] for key, value in dim_reduction_meta.items()
        })
        best_config_by_clustering[clustering_algo_name] = {
            **dim_reduction_meta[best_algo_name],
            "annova": p_value,
            "t-test": t_test_p_value
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
            for cv_id, (cv_data, cv_y) in enumerate(zip(X_cvs, y_cvs)):
                try:
                    cv_data = reduction_algo_wrapper(
                        best_config["reduction_algo_name"],
                        best_config["dim_num"],
                        cv_data, cv_id, CACHE_PATH
                    )
                    labels = clustering_algo(cv_y[external_var_name].nunique(), cv_data)
                    model = CategoricalHMM(n_components=cv_y[external_var_name].nunique()).fit(labels.reshape(-1, 1))
                    hidden_states = model.predict(labels.reshape(-1, 1))
                    scores.append(mutual_info_score(hidden_states, cv_y[external_var_name].values))
                except Exception as e:
                    print(e)
                    scores.append(-1)
            all_mi[external_var_name] = scores
        best_var, p_value, t_test_p_value = find_best_algo(all_mi)
        best_external_var_per_clustering[clustering_algo_name] = {
            "scores": all_mi[best_var],
            "best_var": best_var,
            "anova": p_value,
            "t_test": t_test_p_value
        }
    print(best_external_var_per_clustering)
    with open(f"best_external_var_per_clustering_dynamic.json", "w") as file:
        json.dump(best_external_var_per_clustering, file)
    return best_external_var_per_clustering


def best_config_by(key: str, best_config_by_clustering: Dict):
    clustering_scores = dict()
    for clustering_algo_name, metadata in best_config_by_clustering.items():
        clustering_scores[clustering_algo_name] = metadata[key]
    best_algo_name, p_value, t_test_p_value = find_best_algo(clustering_scores)
    result = {
        "best_algo_name": best_algo_name,
        "annova": p_value,
        "t_test_p_value": t_test_p_value,
        "config": best_config_by_clustering[best_algo_name],
        "all_config": best_config_by_clustering
    }
    print(f"---------final_result_{key}--------")
    print(result)
    with open(f"final_result_{key}.json", "w") as file:
        json.dump(result, file)


if __name__ == '__main__':
    main()
