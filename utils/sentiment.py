import streamlit as st
import re

# Fallback heuristic lexicon for fast & reliable sentiment analysis
POSITIVE_WORDS = {
    'great', 'love', 'awesome', 'excellent', 'helpful', 'fantastic', 'wonderful',
    'easy', 'intuitive', 'perfect', 'clean', 'fast', 'smooth', 'appreciate', 'best',
    'good', 'super', 'pleased', 'valuable', 'nice', 'efficient', 'enjoy'
}

NEGATIVE_WORDS = {
    'crash', 'slow', 'bug', 'error', 'frustrating', 'broken', 'horrible', 'terrible',
    'fail', 'failing', 'useless', 'painful', 'confusing', 'annoying', 'hate', 'difficult',
    'unusable', 'stuck', 'timeout', 'leak', 'worst', 'clunky', 'missing', 'urgently',
    'blocker', 'problem', 'flaw', 'glitch'
}

def rule_based_sentiment(text: str) -> str:
    """Fast, deterministic sentiment scoring fallback."""
    words = re.findall(r'\b\w+\b', str(text).lower())
    pos_score = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_score = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    if neg_score > pos_score:
        return "NEGATIVE"
    elif pos_score > neg_score:
        return "POSITIVE"
    return "NEUTRAL"

@st.cache_resource
def load_sentiment_model():
    """Attempts to load a fast transformer pipeline; returns None on failure."""
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            device=-1
        )
    except Exception:
        return None

def analyze_sentiments(texts, model=None, batch_size=64):
    """
    Analyzes sentiment of a list of feature requests.
    Uses transformer pipeline in batches if available, otherwise fast rule-based fallback.
    Always returns: ['POSITIVE' | 'NEGATIVE' | 'NEUTRAL', ...]
    """
    if model is None:
        return [rule_based_sentiment(t) for t in texts]
        
    try:
        results = []
        truncated = [str(t)[:256] for t in texts]
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            batch_res = model(batch)
            for item in batch_res:
                label = item['label'].upper()
                score = item.get('score', 0.5)
                # Map ambiguous low confidence scores to NEUTRAL
                if score < 0.65:
                    results.append("NEUTRAL")
                else:
                    results.append(label)
        return results
    except Exception:
        # Graceful fallback to rule-based
        return [rule_based_sentiment(t) for t in texts]
