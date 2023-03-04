import json
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, mutual_info_score
from tqdm import tqdm
from karateclub.graph_embedding import Graph2Vec

from src.flows_utils.algorithms import clustering_algorithms, hierarchical_clustering
from src.flows_utils.utils import generate_cvs
from src.static import find_best_algo
from sklearn.metrics import silhouette_score
from sknetwork.embedding import LouvainEmbedding, LouvainNE

DATA_PATH = "data/deezer_ego_nets"
DIMENTIONS_OPTIONS = [10, 50, 100]
NUM_CLUSTERS_OPTIONS = [2, 4, 8, 12, 16, 20]
NUM_OF_CVS = 5
CV_SIZE = 2000
p_value_thr = 0.05


def preprocess_data():
    graphs = []
    with open(os.path.join(DATA_PATH, "deezer_edges.json")) as f:
        graphs_dict: Dict = json.load(f)
    for graph_id, edges in graphs_dict.items():
        curr_graph = nx.Graph()
        for u, v in edges:
            curr_graph.add_edge(u, v)
        graphs.append(curr_graph)
    return graphs
    # adj_matrix = np.zeros((n, n))
    # for i in range(n):
    #     for _, edge in graph_dict[str(i)]:
    #         adj_matrix[i, edge] = 1
    # data = PCA(n_components=0.98).fit_transform(adj_matrix)
    # with open("data.pkl", "wb") as file:
    #     pickle.dump(data, file)
    # return data


def main_flow():
    with open("data.pkl", "rb") as file:
        data = pickle.load(file)
    print(data.shape)
    target = pd.read_csv(os.path.join(DATA_PATH, "deezer_target.csv"))["target"]
    X_cvs, y_cvs = generate_cvs(data, target, NUM_OF_CVS, CV_SIZE)
    best_config_by_clustering = dict()
    for clustering_algo_name in clustering_algorithms.keys():
        scores_by_k = dict()
        for k_clusters in NUM_CLUSTERS_OPTIONS:
            silhouette_scores = []
            mi_scores = []
            scores = []
            for cv_id, (cv_data, cv_y) in enumerate(zip(X_cvs, y_cvs)):
                try:
                    labels = clustering_algorithms[clustering_algo_name](k_clusters, cv_data)
                    sil_score = silhouette_score(cv_data[labels != -1], labels[labels != -1])
                    silhouette_scores.append(sil_score)
                    sil_score = (sil_score + 1) / 2  # normalize between 0 and 1
                    mi_score = normalized_mutual_info_score(labels, cv_y.values)
                    scores.append((sil_score + mi_score) / 2)
                    mi_scores.append(mutual_info_score(labels, cv_y.values))
                except Exception as e:
                    print(e)
                    break
            scores_by_k[k_clusters] = {
                "scores": scores,
                "mi_scores": mi_scores,
                "silhouette_scores": silhouette_scores
            }
        best_k_clusters, p_value, t_test_p_value, msg = find_best_algo({
            key: value["scores"] for key, value in scores_by_k.items()
        })
        best_config_by_clustering[clustering_algo_name] = {
            **scores_by_k[best_k_clusters],
            "best_k_clusters": best_k_clusters,
            "annova": p_value,
            "t_test_p_value": t_test_p_value
        }
        print(f"picking for {clustering_algo_name}: {best_k_clusters}")
    print(best_config_by_clustering)
    # with open("best_config_by_clustering.json") as file:
    #     json.dump(best_config_by_clustering, file)
    clustering_scores = dict()
    for clustering_algo_name, metadata in best_config_by_clustering.items():
        clustering_scores[clustering_algo_name] = metadata["scores"]
    best_k_clusters, p_value, t_test_p_value, msg = find_best_algo(clustering_scores)
    print(f"best_k_clusters: {best_k_clusters}")
    print(f"annova: {p_value}")
    print(f"t test: {t_test_p_value}")


def main():
    graphs = preprocess_data()
    target = pd.read_csv(os.path.join(DATA_PATH, "deezer_target.csv"))["target"].values
    # louvain = LouvainEmbedding()
    louvain = LouvainNE(n_components=10)
    results = []
    max_size = 0
    for graph in tqdm(graphs):
        embedding = louvain.fit_transform(nx.adjacency_matrix(graph))
        # embedding = louvain.fit_transform(nx.adjacency_matrix(graph))
        embedding = embedding.mean(axis=0)
        results.append(embedding)
        max_size = max(max_size, embedding.shape[0])
    # for i in range(len(results)):
    #     num_zeros = max_size - results[i].shape[0]
    #     if num_zeros > 0:
    #         results[i] = np.pad(results[i], (0, num_zeros), 'constant')
    data = np.vstack(results)
    print("doing k means")
    scores = []
    for k in tqdm([2, 6, 12, 20]):
        labels = hierarchical_clustering(k, data)
        scores.append({
            "silhouette_score": silhouette_score(data, labels),
            "mutual_info_score": normalized_mutual_info_score(labels, target)
        })
    print(scores)


if __name__ == '__main__':
    main()
    # main_flow()
