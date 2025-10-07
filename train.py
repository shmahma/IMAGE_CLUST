# train.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

def train_kmeans(embedding_path="embeddings.npy", n_clusters=10):
    data = np.load(embedding_path, allow_pickle=True).item()
    X = data["embeddings"]

    print("🔹 Réduction de dimension avec PCA (50 composantes)...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    print("🔹 Clustering avec KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    np.save("cluster_labels.npy", labels)
    joblib.dump(pca, "pca_model.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")

    print("✅ Modèles et labels sauvegardés (pca_model.pkl, kmeans_model.pkl, cluster_labels.npy)")

if __name__ == "__main__":
    train_kmeans()
