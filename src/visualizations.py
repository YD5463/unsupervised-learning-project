import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples


def hierarchical_clustering_vis(X, ):
    sch.dendrogram(sch.linkage(X[np.random.randint(X.shape[0], size=100), :], method='ward'))
    plt.title("hierarchical clustering - dendrogram")
    plt.plot()


def elbow_method(X: np.ndarray, labels: np.ndarray, n_clusters: int):
    sample_silhouette_values = silhouette_samples(X, labels)
    silhouette_avg = np.mean(sample_silhouette_values)

    y_lower = 10
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
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])


# def reduce_dimension(X):
#     models = [TSNE(n_components=2), Isomap(n_components=2), MDS(n_components=2), SpectralEmbedding(n_components=2)]
#     fig, axs = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 8))
#     fig.tight_layout()
#     for i, model in tqdm(enumerate(models)):
#         X_embedded_tsne = model.fit_transform(X)
#         axs[i].scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], s=40, cmap='viridis')
#         axs[i].set_title(f"{model} dimensionality reduction")
#     plt.show()
