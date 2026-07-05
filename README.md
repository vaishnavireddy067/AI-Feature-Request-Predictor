# AI Feature Request Prioritizer

A modern AI-powered web application built with Python and Streamlit to help product managers automatically analyze thousands of customer feature requests and identify the most important product improvements.

## Features
- **Semantic Similarity**: Uses `all-MiniLM-L6-v2` to embed requests and FAISS to detect duplicates.
- **Clustering**: Organizes requests into categories (e.g., Dashboard, Payments, Security) using KMeans.
- **Summarization**: Generates concise summaries of clusters using `facebook/bart-large-cnn`.
- **Keyword Extraction**: Identifies top keywords in categories via `KeyBERT`.
- **Ranking**: Ranks features by popularity, frequency, and semantic similarity.
- **Analytics Dashboard**: Interactive charts, WordClouds, and semantic search.

## Installation

1. Clone or download this repository.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Generate sample data (if not already provided):
   ```bash
   python data/generate_sample_data.py
   ```

## Usage

Start the Streamlit application:
```bash
streamlit run app.py
```
Upload your feature request CSV or use the provided sample.
