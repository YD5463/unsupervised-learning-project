import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, mutual_info_score
from tqdm import tqdm

from src.flows_utils.algorithms import dim_reduction_algorithms, clustering_algorithms, anomaly_detection_algorithms
from scipy.stats import f_oneway
from scipy.stats import ttest_rel

P_VALUE_THR = 0.05


def get_silhouette_scores(
        X_cvs: List[np.ndarray],
        clustering_algo_name: str, reduction_algo_name: str,
        dim_num: int, k_clusters: int
) -> List[float]:
    scores = []
    for cv_id, cv_data in enumerate(X_cvs):
        try:
            cv_data = reduction_algo_wrapper(reduction_algo_name, dim_num, cv_data, cv_id)
            print(f"doing {clustering_algo_name} for {cv_data.shape} by {reduction_algo_name}")
            labels = clustering_algorithms[clustering_algo_name](k_clusters, cv_data)
            # print(f"anomaly: {labels[labels == -1].size / labels.size}")
            scores.append(silhouette_score(cv_data[labels != -1], labels[labels != -1]))
        except Exception as e:
            print(e)
            break
    return scores


def reduction_algo_wrapper(reduction_algo_name: str, dim_num: int, cv_data: np.ndarray, cv_id: int, CACHE_PATH) -> np.ndarray:
    reduction_algo = dim_reduction_algorithms[reduction_algo_name]
    if reduction_algo is None:
        return cv_data
    cache_file = os.path.join(CACHE_PATH, f"{reduction_algo_name}-{dim_num}-{cv_id}.pkl")
    if Path(cache_file).is_file():
        with open(cache_file, "rb") as file:
            return pickle.load(file)
    cv_data = reduction_algo(dim_num).fit_transform(cv_data)
    with open(cache_file, "wb") as file:
        pickle.dump(cv_data, file)
    return cv_data


def generate_cvs(X: np.ndarray, y: pd.DataFrame, num_of_cvs, cv_size) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
    X_cvs = []
    y_cvs = []
    for i in range(num_of_cvs):
        rows = np.random.randint(X.shape[0], size=cv_size)
        X_cvs.append(X[rows, :])
        y_cvs.append(y.iloc[rows])
    return X_cvs, y_cvs


def find_best_algo(scores_mapping: Dict[Any, List[float]]) -> Tuple[Any, float, float]:
    _, p_value = f_oneway(*list(scores_mapping.values()))
    best_algo = random.choice(list(scores_mapping.keys()))
    t_test_p_value = -1
    print(f"annova value: {p_value}")
    if p_value < P_VALUE_THR:
        sorted_scores = sorted(
            scores_mapping,
            key=lambda key: np.mean(scores_mapping[key]),
            reverse=True
        )
        candidate1, candidate2 = sorted_scores[:2]
        _, t_test_p_value = ttest_rel(
            scores_mapping[candidate1],
            scores_mapping[candidate2]
        )
        print(f"t_test value: {t_test_p_value}")
        best_algo = sorted_scores[0]
        if t_test_p_value >= P_VALUE_THR:
            print(f"followed by t-test: algorithms {candidate1}, {candidate2} are the same")
    else:
        print(f"followed by annova: algorithms {scores_mapping.keys()} are the same")
    return best_algo, p_value, t_test_p_value


def external_var_to_anomalies(X_cvs, y_cvs, external_vars):
    results = list()
    for anomaly_algo_name, anomaly_algo in tqdm(anomaly_detection_algorithms.items()):
        if anomaly_algo is None:
            continue
        scores = defaultdict(list)
        for X, y in zip(X_cvs, y_cvs):
            labels = anomaly_algo.fit_predict(X)
            for external_var_name in external_vars:
                scores[external_var_name].append(
                    mutual_info_score(
                        labels,
                        y[external_var_name]
                    )
                )

        for external_var_name, external_var_scores in scores.items():
            results.append({
                "algo_name": anomaly_algo_name,
                "external_var": external_var_name,
                "MI": np.mean(external_var_scores)
            })

    return pd.DataFrame(results)
