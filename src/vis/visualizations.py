import json

import pandas as pd
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import seaborn as sns

sns.set_theme()


def hierarchical_clustering_vis(X, ):
    sch.dendrogram(sch.linkage(X[np.random.randint(X.shape[0], size=100), :], method='ward'))
    plt.title("hierarchical clustering - dendrogram")
    plt.plot()


def elbow_method(X: np.ndarray, labels: np.ndarray, n_clusters: int):
    sample_silhouette_values = silhouette_samples(X, labels)
    y_lower = 10
    plt.figure(figsize=(15, 8))
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # TODO: legend instead of text
        # plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    plt.yticks([])
    plt.savefig("elbow.png")


# def reduce_dimension(X):
#     models = [TSNE(n_components=2), Isomap(n_components=2), MDS(n_components=2), SpectralEmbedding(n_components=2)]
#     fig, axs = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 8))
#     fig.tight_layout()
#     for i, model in tqdm(enumerate(models)):
#         X_embedded_tsne = model.fit_transform(X)
#         axs[i].scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], s=40, cmap='viridis')
#         axs[i].set_title(f"{model} dimensionality reduction")
#     plt.show()

def anomaly_external_var_to_mi(df):
    sns.set(rc={'figure.figsize':(6,6)})
    g = sns.catplot(
        data=df, kind="bar",
        x="external_var", y="MI", hue="algo_name",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("External Variable", "Mutual Information")
    g.legend.set_title("")
    plt.savefig("anomaly_external_var_to_mi.png")


def plot_silhouette():
    with open("./reports/without_anomaly_algo/silout_per_clustreing.json", "r") as file:
        without_anomaly_algo_scores = json.load(file)
    with open("./reports/with_anomaly_algo/sillout_pre_clustering.json", "r") as file:
        with_anomaly_algo_scores = json.load(file)
    rows = []
    for key, scores in without_anomaly_algo_scores.items():
        for score in scores:
            rows.append({
                "Clustering Algorithm": key.replace("_", " ").title(),
                "score": score,
                "With Anomaly Filtering": "No"
            })
    for key, scores in with_anomaly_algo_scores.items():
        for score in scores:
            rows.append({
                "Clustering Algorithm": key.replace("_", " ").title(),
                "score": score,
                "With Anomaly Filtering": "Yes"
            })
    df = pd.DataFrame(rows)
    plt.figure(figsize=(15, 8))
    g = sns.barplot(
        data=df, x="Clustering Algorithm", y="score",
        hue="With Anomaly Filtering", errorbar="sd",  palette="dark", alpha=.6,
    )
    g.set_ylabel("Silhouette Scores")
    g.set_xlabel("")
    g.set_title("Silhouette Scores By Algorithm")
    plt.savefig("static_silhouettes.png")


def plot_mi_dynamic():
    with open("./reports/dynamic/final_result_weighted_scores.json", "r") as file:
        final_result_weighted_scores = json.load(file)
    final_result_weighted_scores = final_result_weighted_scores["all_config"]
    rows = []
    for key, scores in final_result_weighted_scores.items():
        for score in scores["weighted_scores"]:
            rows.append({
                "Clustering Algorithm": key.replace("_", " ").title(),
                "score": score,
            })
    df = pd.DataFrame(rows)
    plt.figure(figsize=(15, 8))
    g = sns.barplot(
        data=df, x="Clustering Algorithm", y="score",
        errorbar="sd",  palette="dark", alpha=.6,
    )
    g.set_ylabel("MI Scores")
    g.set_xlabel("")
    g.set_title("MI Scores By Algorithm")
    plt.savefig("mi_dynamic.png")
