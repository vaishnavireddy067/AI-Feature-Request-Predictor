from keybert import KeyBERT
import streamlit as st

@st.cache_resource
def load_keyword_model():
    """
    Load KeyBERT model.
    """
    return KeyBERT('all-MiniLM-L6-v2')

def extract_keywords(text, kw_model, top_n=5):
    """
    Extract top keywords from text using KeyBERT.
    """
    if not text or len(text.strip()) == 0:
        return []
        
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=top_n
    )
    
    # Return just the keyword strings, not the scores
    return [kw[0] for kw in keywords]
