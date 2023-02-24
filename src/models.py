import pickle
import random
from typing import Tuple, Dict, Any, List
import numpy as np
import os
import pandas as pd
from src.algorithms import clustering_algorithms, dim_reduction_algorithms, anomaly_detection_algorithms
from sklearn.metrics import mutual_info_score
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

OUTPUT_PATH = "../output/"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
CACHE_PATH = "../cache/"
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

# Config
# random_state = 0
data_path = "../data/fma_metadata"
dimentions_options = [10, 50, 100, 200]
num_clusters_options = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
num_of_cvs = 7
cv_size = 10000
p_value_thr = 0.05
external_vars = ["('track', 'genre_top')", "('track', 'license')", "('album', 'type')"]


def load_data() -> Tuple[np.ndarray, pd.DataFrame]:
    cache_file = f"{OUTPUT_PATH}/clean_static_data.csv"
    if Path(cache_file).is_file():
        df = pd.read_csv(cache_file)
    else:
        features = pd.read_csv(os.path.join(data_path, "features.csv"), index_col=0, header=[0, 1, 2])
        tracks = pd.read_csv(os.path.join(data_path, "tracks.csv"), index_col=0, header=[0, 1])
        features.columns = features.columns.map('_'.join)
        df = pd.concat([features, tracks[external_vars]], axis=1)
        df = df.dropna()
        df.to_csv(cache_file)
    X, y = df.drop(external_vars, axis=1), df[external_vars]
    for column in external_vars:
        y[column] = y[column].astype('category').cat.codes
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    return X, y


def generate_cvs(X: np.ndarray, y: pd.DataFrame) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
    X_cvs = []
    y_cvs = []
    for i in range(num_of_cvs):
        rows = np.random.randint(X.shape[0], size=cv_size)
        X_cvs.append(X[rows, :].values)
        y_cvs.append(y.iloc[rows])
    return X_cvs, y_cvs


def reduction_algo_wrapper(reduction_algo_name: str, dim_num: int, cv_data: np.ndarray, cv_id: int) -> np.ndarray:
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


def find_best_algo(scores_mapping: Dict[str, List[float]], random_when_same: bool = True) -> str:
    _, p_value = f_oneway(*list(scores_mapping.values()))
    best_algo = random.choice(list(scores_mapping.values()))
    if p_value < p_value_thr:
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
        best_algo_name: str = sorted_scores[0]
        best_algo = scores_mapping[best_algo_name]
        if t_test_p_value >= p_value_thr:
            print(f"followed by t-test: algorithms {candidate1}, {candidate2} are the same")
            if not random_when_same:
                return f"{candidate1, candidate2}"
    else:
        print(f"followed by annova: algorithms {best_algo.keys()} are the same")
        if not random_when_same:
            return ""
    return best_algo


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
            print(f"anomaly: {labels[labels == -1].size / labels.size}")
            scores.append(silhouette_score(cv_data[labels != -1], labels[labels != -1]))
        except Exception as e:
            print(e)
            scores.append(-1)
    return scores


def main_flow(X_cvs: List[np.ndarray]) -> Dict[str, Dict[str, Any]]:
    best_config_by_clustering = dict()
    for clustering_algo_name in clustering_algorithms.keys():
        dim_reduction_scores = dict()
        dim_reduction_meta: Dict[str, Dict[str, Any]] = dict()
        for reduction_algo_name in tqdm(dim_reduction_algorithms.keys()):
            max_score = float("-inf")
            for dim_num in dimentions_options:
                for k_clusters in num_clusters_options:
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
        best_algo_name = find_best_algo(dim_reduction_scores)
        best_config_by_clustering[clustering_algo_name] = dim_reduction_meta[best_algo_name]
        print(f"picking for {clustering_algo_name}: {best_config_by_clustering[clustering_algo_name]}")
    return best_config_by_clustering


def second_flow(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame],
                best_config_by_clustering: Dict[str, Dict[str, Any]]):
    # hidden variables with the best clustering and dim_reduction
    best_cluster_algo_per_external_var = dict()
    for external_var_name in external_vars:
        all_mi = dict()
        for clustering_algo_name, clustering_algo in clustering_algorithms.items():
            scores = []
            best_config = best_config_by_clustering[clustering_algo_name]
            for cv_id, (cv_data, cv_y_true) in enumerate(zip(X_cvs, y_cvs)):
                try:
                    cv_data = reduction_algo_wrapper(
                        best_config["reduction_algo_name"],
                        best_config["max_dim_num"],
                        cv_data, cv_id
                    )
                    labels = clustering_algo(best_config["max_cluster_num"], cv_data)
                    scores.append(mutual_info_score(labels, cv_y_true[external_var_name].values))
                except Exception as e:
                    print(e)
                    scores.append(-1)
            all_mi[clustering_algo_name] = scores
        best_algo_name = find_best_algo(all_mi)
        best_cluster_algo_per_external_var[external_var_name] = {
            "algo_name": best_algo_name,
            "scores": all_mi[best_algo_name]
        }
    print(best_cluster_algo_per_external_var)
    with open(f"{OUTPUT_PATH}/best_cluster_algo_per_external_var.pkl", "wb") as file:
        pickle.dump(best_cluster_algo_per_external_var, file)
    return best_cluster_algo_per_external_var


def third_flow(X_cvs: List[np.ndarray], y_cvs: List[pd.DataFrame],
               best_config_by_clustering: Dict[str, Dict[str, Any]]):
    best_external_var_per_clustering = dict()
    for clustering_algo_name, clustering_algo in clustering_algorithms.items():
        all_mi = dict()
        for external_var_name in external_vars:
            scores = []
            best_config = best_config_by_clustering[clustering_algo_name]
            for cv_id, (cv_data, cv_y_true) in enumerate(zip(X_cvs, y_cvs)):
                try:
                    cv_data = reduction_algo_wrapper(
                        best_config["reduction_algo_name"],
                        best_config["max_dim_num"],
                        cv_data, cv_id
                    )
                    labels = clustering_algo(cv_y_true[external_var_name].nunique(), cv_data)
                    scores.append(mutual_info_score(labels, cv_y_true[external_var_name].values))
                except Exception as e:
                    print(e)
                    scores.append(-1)
            all_mi[external_var_name] = scores
        best_external_var_per_clustering[clustering_algo_name] = find_best_algo(all_mi, random_when_same=False)
    print(best_external_var_per_clustering)
    with open(f"{OUTPUT_PATH}/best_external_var_per_clustering.pkl", "wb") as file:
        pickle.dump(best_external_var_per_clustering, file)
    return best_external_var_per_clustering
