import json
import os
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import normalized_mutual_info_score

from src.flows_utils.algorithms import dim_reduction_algorithms, clustering_algorithms
from src.flows_utils.utils import reduction_algo_wrapper, find_best_algo
from src.static import dimentions_options, num_clusters_options
from pathlib import Path


CACHE_PATH = "cache-dynamic/"
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)
EXTERNAL_VARS = ["gas_type", "concentration"]
DIMENTIONS_OPTIONS = [10, 50, 100]
NUM_CLUSTERS_OPTIONS = [2, 4, 8, 12, 16, 20]
DATA_PATH = "data/driftdataset"


def load_dynamic_dataset():
    dfs = []
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
            dfs.append(pd.DataFrame(df_rows))
    return dfs


def main():
    dfs = load_dynamic_dataset()
    best_config_by_clustering = dict()
    for clustering_algo_name in clustering_algorithms.keys():
        dim_reduction_meta: Dict[str, Dict[str, Any]] = dict()
        for reduction_algo_name in dim_reduction_algorithms.keys():
            max_score = float("-inf")
            for dim_num in dimentions_options:
                for k_clusters in num_clusters_options:
                    scores = defaultdict(list)
                    for cv_id, df in enumerate(dfs):
                        X = df.drop(EXTERNAL_VARS, axis=1).values
                        y = df[EXTERNAL_VARS]
                        y["concentration"] = y["concentration"].apply(lambda val: int(float(val)))
                        try:
                            cv_data = reduction_algo_wrapper(reduction_algo_name, dim_num, X, cv_id, CACHE_PATH)
                            print(f"doing {clustering_algo_name} for {cv_data.shape} by {reduction_algo_name}")
                            labels = clustering_algorithms[clustering_algo_name](k_clusters, cv_data)
                            model = CategoricalHMM(n_components=y["gas_type"].nunique()).fit(labels.reshape(-1, 1))
                            hidden_states_gas_type = model.predict(labels.reshape(-1, 1))
                            mi_gas_type_mi = normalized_mutual_info_score(
                                y["gas_type"].values,
                                hidden_states_gas_type
                            )
                            model = CategoricalHMM(n_components=y["concentration"].nunique()).fit(labels.reshape(-1, 1))
                            hidden_states_concentration = model.predict(labels.reshape(-1, 1))
                            mi_concentration = normalized_mutual_info_score(
                                y["gas_type"].values,
                                hidden_states_concentration
                            )
                            scores["mi_concentration_scores"].append(mi_concentration)
                            scores["mi_gas_type_scores"].append(mi_gas_type_mi)
                            scores["weighted_scores"].append(
                                (mi_concentration + mi_gas_type_mi) / 2
                            )
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
    clustering_scores = dict()
    for clustering_algo_name, metadata in best_config_by_clustering.items():
        clustering_scores[clustering_algo_name] = metadata["weighted_scores"]
    best_algo_name, p_value, t_test_p_value = find_best_algo(clustering_scores)
    print(best_config_by_clustering)
    result = {
        "best_algo_name": best_algo_name,
        "annova": p_value,
        "t_test_p_value": t_test_p_value,
        "config": best_config_by_clustering[best_algo_name],
        "all_config": best_config_by_clustering
    }
    print(result)
    with open("final_result.json", "w") as file:
        json.dump(result, file)


if __name__ == '__main__':
    main()
