import numpy as np
from keras import Model, Input
from keras.legacy_tf_layers.core import Dense
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, MDS
from fcmeans import FCM
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import hdbscan
import scipy.cluster.hierarchy as sch


def k_mean(k, data: np.ndarray):
    model = KMeans(k, max_iter=1000)
    labels = model.fit_predict(data)
    return labels


def fuzzy_c_means(k, data: np.ndarray):
    model = FCM(n_clusters=k, n_jobs=8)
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
    model = SpectralClustering(n_clusters=k, assign_labels='discretize')
    labels = model.fit_predict(data)
    return labels


def hierarchical_dbscan(k: int, data: np.ndarray):
    model = hdbscan.HDBSCAN()
    model.fit(data)
    return model.labels_


def get_best_dbscan_params(clean_X: np.ndarray, min_samples: int):
    range_eps = np.linspace(0.1, 10, 20)
    scores = []
    for eps in range_eps:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        good_labels = model.fit_predict(clean_X)
        noisy_data_count = len(good_labels[good_labels == -1])
        if noisy_data_count > good_labels.shape[0] * 0.5 or len(np.unique(good_labels)) < 3:
            scores.append(0)
            continue
        print(f"noisy data count with eps={eps}: {noisy_data_count}")
        try:
            score = silhouette_score(clean_X, good_labels)
        except Exception as e:
            print(e)
            score = 0
        scores.append(score)
    plt.plot(range_eps, scores)
    plt.show()
    return range_eps[np.argmax(scores)]


def visualize_hierarchical_clustering(data: np.ndarray):
    sch.dendrogram(sch.linkage(data, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()


clustering_algorithms = {
    "k_mean": k_mean,
    "fuzzy_c_means": fuzzy_c_means,
    "gaussian_mixture": gaussian_mixture,
    "hierarchical_clustering": hierarchical_clustering,
    "birch": birch,
    "spectral_clustering": spectral_clustering,
    "hierarchical_dbscan": hierarchical_dbscan,
}


# def autoencoder_dim_reduction():
#     input_img = Input(shape=(img.width,))
#     encoded1 = Dense(128, activation='relu')(input_img)
#     encoded2 = Dense(reduced_pixel, activation='relu')(encoded1)
#     decoded1 = Dense(128, activation='relu')(encoded2)
#     decoded2 = Dense(img.width, activation=None)(decoded1)
#     autoencoder = Model(input_img, decoded2)
#     autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
#     autoencoder.fit(X, X,
#                     epochs=500,
#                     batch_size=16,
#                     shuffle=True)
#     # Encoder
#     encoder = Model(input_img, encoded2)
#     # Decoder
#     decoder = Model(input_img, decoded2)
#     encoded_imgs = encoder.predict(X)
#     decoded_imgs = decoder.predict(X)


dim_reduction_algorithms = {
    "TSNE": lambda k: TSNE(n_components=k, method="exact", init="random"),
    "Isomap": lambda k: Isomap(n_components=k),
    "MDS": lambda k: MDS(n_components=k),
    "SpectralEmbedding": lambda k: SpectralEmbedding(n_components=k),
    "PCA": lambda k: PCA(n_components=k),
    "FastICA": lambda k: FastICA(n_components=k),
    "without_reduction": None
}

anomaly_detection_algorithms = {
    "OneClassSVM": OneClassSVM(),
    "IsolationForest": IsolationForest(n_estimators=500),
    "DBSCAN": DBSCAN(eps=0.4, min_samples=10)
}
