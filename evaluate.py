# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import tensorflow as tf

def evaluate_models():
    data = np.load("embeddings.npy", allow_pickle=True).item()
    X = data["embeddings"]
    y_true = data["labels"].flatten()
    labels = np.load("cluster_labels.npy")

    pca = joblib.load("pca_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    X_vis = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap="tab10", s=10)
    plt.title("Clusters d’images CIFAR-10 (ResNet50 + KMeans)")
    plt.show()

    score = silhouette_score(pca.transform(X), labels)
    print(f"📈 Silhouette Score : {score:.4f}")

if __name__ == "__main__":
    evaluate_models()
