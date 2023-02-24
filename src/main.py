import numpy as np

from src.autoencoder import auto_encoder_fit_predict
from src.models import load_data, main_flow, second_flow, third_flow, generate_cvs


def main():
    X, y = load_data()
    rows = np.random.randint(X.shape[0], size=100)
    X, y = X[rows, :], y.iloc[rows]
    X_cvs, y_cvs = generate_cvs(X, y)
    best_config_by_clustering = main_flow(X_cvs)
    best_cluster_algo_per_external_var = second_flow(X_cvs, y_cvs, best_config_by_clustering)
    best_external_var_per_clustering = third_flow(X_cvs, y_cvs, best_config_by_clustering)
    print("--------best_config_by_clustering------------")
    print(best_config_by_clustering)
    print("--------best_cluster_algo_per_external_var------------")
    print(best_cluster_algo_per_external_var)
    print("--------best_external_var_per_clustering------------")
    print(best_external_var_per_clustering)


def main2():
    X, y = load_data()
    rows = np.random.randint(X.shape[0], size=100)
    X, y = X[rows, :], y.iloc[rows]
    out = auto_encoder_fit_predict(10, X)


if __name__ == '__main__':
    main()
