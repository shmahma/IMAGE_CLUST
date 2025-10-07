# ImageClust

## Overview
This project performs **image clustering** on the CIFAR-10 dataset using deep learning embeddings and unsupervised learning.  
A `ResNet50` model is used to extract image embeddings, which are then clustered using **KMeans** after optional PCA dimensionality reduction.

Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  

A Python pipeline allows easy extraction of embeddings, training of clustering models, and visualization of clusters.

---

## Features

### Image Embeddings
- **Model used:** ResNet50 (`keras.applications.ResNet50`)  
- **Layer used:** `avg_pool`  
- **Task:** Feature extraction from images

### Clustering
- **Algorithm:** KMeans (unsupervised clustering)  
- **Optional Dimensionality Reduction:** PCA (50 components)  
- **Labels:** Cluster assignments for each image

### Visualization & Evaluation
- **Visuals:** 2D scatter plot of clusters  
- **Metrics:** Silhouette Score

---

## Installation
Clone the repository:

```bash
git clone https://github.com/votre-utilisateur/ImageClust.git
cd ImageClust

pip install -r requirements.txt

python preprocess.py
python train.py
python evaluate.py

