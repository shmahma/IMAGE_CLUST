# preprocess.py
import numpy as np
from tensorflow.keras.applications import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
from tqdm import tqdm

def extract_embeddings(batch_size=64, save_path="embeddings.npy"):
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")

    x_resized = tf.image.resize(x_train, (224, 224))

    base_model = ResNet50(weights="imagenet", include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

    embeddings = []
    for i in tqdm(range(0, len(x_resized), batch_size)):
        batch = x_resized[i:i+batch_size].numpy()
        batch = preprocess_input(batch)
        emb = model.predict(batch, verbose=0)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    np.save(save_path, {"embeddings": embeddings, "labels": y_train})
    print(f"✅ Embeddings sauvegardés dans {save_path}")

if __name__ == "__main__":
    extract_embeddings()
