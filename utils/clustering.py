from sklearn.cluster import KMeans
import streamlit as st
import numpy as np

def perform_clustering(embeddings, n_clusters=10):
    """
    Clusters embeddings into n_clusters using KMeans.
    Safely clamps n_clusters if dataset is small.
    """
    n_samples = len(embeddings)
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int)
        
    actual_clusters = max(1, min(n_clusters, n_samples))
    if actual_clusters == 1:
        return np.zeros(n_samples, dtype=int)
        
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels
