import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, BisectingKMeans
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, MDS
from fcmeans import FCM
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.manifold import LocallyLinearEmbedding
from umap import UMAP


def k_mean(k, data: np.ndarray):
    model = KMeans(k, max_iter=1000)
    labels = model.fit_predict(data)
    return labels


def fuzzy_c_means(k, data: np.ndarray):
    model = FCM(n_clusters=k, n_jobs=-1)
    model.fit(data)
    return model.predict(data)


def gaussian_mixture(k: int, data: np.ndarray):
    model = GaussianMixture(k, max_iter=3000)
    labels = model.fit_predict(data)
    return labels


def hierarchical_clustering(k: int, data: np.ndarray):
    model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    labels = model.fit_predict(data)
    return labels


def birch(k: int, data: np.ndarray):
    model = Birch(n_clusters=k)
    labels = model.fit_predict(data)
    return labels


def spectral_clustering(k: int, data: np.ndarray):
    model = SpectralClustering(n_clusters=k, assign_labels='discretize', n_jobs=-1)
    labels = model.fit_predict(data)
    return labels


def dbscan(k: int, data: np.ndarray):
    model = DBSCAN(eps=((k / 100) ** 2) * data.shape[1], min_samples=5, n_jobs=-1)
    labels = model.fit_predict(data)
    return labels


def bisecting_kmeans(k: int, data: np.ndarray):
    model = BisectingKMeans(n_clusters=k)
    labels = model.fit_predict(data)
    return labels


clustering_algorithms = {
    "k_mean": k_mean,
    "fuzzy_c_means": fuzzy_c_means,
    "gaussian_mixture": gaussian_mixture,
    "hierarchical_clustering": hierarchical_clustering,
    "birch": birch,
    "spectral_clustering": spectral_clustering,
    "dbscan": dbscan,
    "bisecting_kmeans": bisecting_kmeans
}

dim_reduction_algorithms = {
    "MDS": lambda k: MDS(n_components=k, n_jobs=-1, max_iter=300),
    "PCA": lambda k: PCA(n_components=k),
    "FastICA": lambda k: FastICA(n_components=k, max_iter=200),
    "without_reduction": None,
    "Isomap": lambda k: Isomap(n_components=k, n_jobs=-1, max_iter=200),
    "SpectralEmbedding": lambda k: SpectralEmbedding(n_components=k, n_jobs=-1),
    "LLE": lambda k: LocallyLinearEmbedding(n_components=k, n_jobs=-1),
    "UMAP": lambda k: UMAP(n_neighbors=100, n_components=k, n_epochs=1000, init='spectral', low_memory=False, verbose=False)
    # "TSNE": lambda k: TSNE(n_components=k, method="exact", init="random", n_jobs=-1, n_iter=250),
}

anomaly_detection_algorithms = {
    "without_anomaly": None,
    "OneClassSVM": OneClassSVM(kernel="rbf", nu=0.01, gamma='scale'),
    "IsolationForest": IsolationForest(random_state=0, n_jobs=-1, n_estimators=500, max_samples=256),
    "DBSCAN": DBSCAN(eps=5, min_samples=5, n_jobs=-1)
}
