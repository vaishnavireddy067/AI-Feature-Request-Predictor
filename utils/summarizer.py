from sentence_transformers import SentenceTransformer, util
import streamlit as st
import random
import numpy as np

@st.cache_resource
def load_summarizer():
    """
    Load the sentence transformer model for extractive summarization.
    Using 'all-MiniLM-L6-v2' to ensure blazing fast processing (under 30s).
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def summarize_cluster(texts, model, max_samples=30):
    """
    Generates an extractive summary for a cluster of texts.
    Finds the most representative request (closest to the centroid of the cluster).
    """
    if not texts:
        return "No data to summarize."
        
    # Take a sample to keep it lightning fast
    sample_texts = random.sample(texts, min(len(texts), max_samples))
    
    try:
        # Get embeddings for the sample texts
        embeddings = model.encode(sample_texts)
        
        # Calculate the cluster's centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Find the text whose embedding is closest to the centroid
        cos_scores = util.cos_sim(centroid, embeddings)[0]
        best_idx = np.argmax(cos_scores).item()
        
        best_text = sample_texts[best_idx]
        return f"Representative Request: \"{best_text}\""
    except Exception as e:
        return f"Could not generate summary: {str(e)}"
