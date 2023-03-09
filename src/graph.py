import json
import os
import warnings
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
from scipy.sparse import csgraph
import seaborn as sns
from sklearn.metrics import silhouette_score

from src.flows_utils.algorithms import anomaly_detection_algorithms, clustering_algorithms
from src.flows_utils.utils import find_best_algo, generate_cvs

warnings.filterwarnings("ignore")
DATA_PATH = "data/deezer_ego_nets"
NUM_CLUSTERS_OPTIONS = [2, 10, 50, 100, 500]
NUM_OF_CVS = 5
CV_SIZE = 2000
p_value_thr = 0.05


def load_graph():
    graphs = []
    with open(os.path.join(DATA_PATH, "deezer_edges.json")) as f:
        graphs_dict: Dict = json.load(f)
    for graph_id, edges in graphs_dict.items():
        curr_graph = nx.Graph()
        for u, v in edges:
            curr_graph.add_edge(u, v)
        graphs.append(curr_graph)
    return graphs


def preprocess() -> np.ndarray:
    graphs = load_graph()
    results = []
    max_size = 0
    pca = PCA(n_components=2)
    for graph in tqdm(graphs):
        laplacian = csgraph.laplacian(nx.adjacency_matrix(graph), normed=True).toarray()
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        eigenvalues = eigenvalues.real
        eigvals_norm = np.linalg.norm(eigenvalues)
        eigenvalues = eigenvalues / eigvals_norm
        eigenvalues = np.sort(eigenvalues)
        results.append(eigenvalues)
        max_size = max(max_size, eigenvalues.shape[0])
    for i in range(len(results)):
        num_zeros = max_size - results[i].shape[0]
        if num_zeros > 0:
            results[i] = np.pad(results[i], (0, num_zeros), 'constant')
    data = np.vstack(results)
    data = pca.fit_transform(data)
    return data


def main():
    target: pd.DataFrame = pd.read_csv(os.path.join(DATA_PATH, "deezer_target.csv"))
    data = preprocess()
    X_cvs, y_cvs = generate_cvs(data, target, NUM_OF_CVS, CV_SIZE)
    anomaly_scores = dict()
    results = []
    for anomaly_algo_name, anomaly_algo in anomaly_detection_algorithms.items():
        print(f"-------------{anomaly_algo_name}-----------")
        best_config_by_clustering = dict()
        if anomaly_algo:
            anomaly_labels = anomaly_algo.fit_predict(data)
            data = data[anomaly_labels != -1]
            anomaly_labels[anomaly_labels == -1] = 0
            anomaly_scores["anomaly_algo_name"] = normalized_mutual_info_score(anomaly_labels, target)

        for clustering_algo_name, clustering_algo in clustering_algorithms.items():
            print(f"------------{clustering_algo_name}-----------")
            scores = defaultdict(list)
            max_score = float("-inf")
            for k in tqdm(NUM_CLUSTERS_OPTIONS):
                for cv_data, cv_y_true in zip(X_cvs, y_cvs):
                    try:
                        labels = clustering_algo(k, cv_data)
                        sil_score = silhouette_score(cv_data, labels)
                        mi_score = normalized_mutual_info_score(labels, cv_y_true["target"].values)
                        scores["silhouette_scores"].append(sil_score)
                        scores["mutual_information_scores"].append(mi_score)
                        scores["weighted_score"].append((mi_score + (sil_score + 1) / 2))
                    except Exception as e:
                        print(e)
                        break
                avg_score = np.mean(scores["weighted_scores"])
                if avg_score > max_score:
                    max_score = avg_score
                    best_config_by_clustering[clustering_algo_name] = {
                        "avg_score": avg_score,
                        "k": k,
                        **scores
                    }
        best_algo, p_value, t_test_p_value, msg = find_best_algo({
            key: value["weighted_scores"] for key, value in best_config_by_clustering.items()
        })
        print(best_config_by_clustering)
        results.append({
            "algo_name": anomaly_algo_name,
            "best_config": best_config_by_clustering[best_algo],
            "annova": p_value,
            "t-test": t_test_p_value,
            "msg": msg
        })
    print("------------------------")
    print(results)
    df = pd.DataFrame(results)
    df.to_csv("results.csv")
    plot_results(df, "weighted_scores")
    plot_results(df, "mutual_information_scores")
    plot_results(df, "silhouette_scores")


def plot_results(df: pd.DataFrame, metric_name: str):
    plt.figure(figsize=(15, 8))
    g = sns.barplot(
        data=df, x="algo_name", y=metric_name,
        hue="Anomaly Detector", errorbar="sd",  palette="dark", alpha=.6,
    )
    g.set_ylabel(metric_name.replace("_", " ").title())
    g.set_xlabel("")
    plt.savefig(f"graph_{metric_name}.png")


if __name__ == '__main__':
    main()

