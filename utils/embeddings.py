from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np

@st.cache_resource
def load_embedding_model():
    """
    Load the sentence transformer model.
    Cached by Streamlit to avoid reloading on every interaction.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts, model):
    """
    Generates embeddings for a list of texts.
    Returns a numpy array of embeddings.
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype('float32')
