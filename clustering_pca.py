# clustering_pca.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from preprocessing import load_and_preprocess


def run_pca_kmeans(path, n_clusters=3):
    df = load_and_preprocess(path)

    X = df.drop(columns=["Data Value"])

    # PCA to 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    plt.title("K-Means Clusters (PCA Reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    return pca_result, clusters


if __name__ == "__main__":
    run_pca_kmeans("../data/Air_Quality_Cleaned_Data.csv")
