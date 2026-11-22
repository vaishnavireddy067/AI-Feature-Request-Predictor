# 🚀 AI Feature Request Prioritizer

An enterprise-grade, AI-powered product discovery and roadmap intelligence web application built with Python and Streamlit. It automates customer feedback analysis, detects duplicate requests using dense semantic vector search, clusters related feedback into cohesive strategic themes, calculates explainable **Smart Priority Scores (RICE)**, forecasts demand velocity momentum, breaks down customer segments, and auto-generates executive reports and roadmaps.

---

## 🌟 Key Features

### 1. 🎯 Smart Priority Score (RICE Framework)
- Evaluates every feature request cluster across **Reach**, **Impact**, **Confidence**, and **Effort**:
  $$\text{RICE Score} = \frac{\text{Reach} \times \text{Impact} \times \text{Confidence}}{\text{Effort}}$$
- **Reach**: User exposure weighted by request volume, upvotes, and customer tiers.
- **Impact**: Quantified from customer sentiment friction (severe pain points) and enterprise revenue exposure.
- **Confidence**: Statistical confidence based on feedback density and keyword cohesion.
- **Effort**: Domain-based engineering complexity (1-10 scale), with real-time interactive sliders.
- Interactive **Plotly Impact vs. Effort Matrix** showing *Quick Wins*, *Strategic Bets*, *Fill-ins*, and *Time Sinks*.

### 2. 💡 Evidence-Backed AI Insights
- Transparent AI decision rationales explaining *why* a feature was recommended (e.g. churn risk, enterprise upsell, fast ROI).
- Real quantitative metrics (exact feedback counts, share percentages, and segment breakdown).
- **Verbatim Customer Quotes**: Representative user feedback highlighting real customer pain points.
- Actionable recommendations customized for product managers and engineering leads.

### 3. 🗺️ AI Roadmap Generator (Now / Next / Later)
- Automatically routes feature themes into execution horizons:
  - 🔥 **NOW**: Immediate sprint candidates (Critical friction points, high RICE, low effort).
  - ⭐ **NEXT**: Planned quarterly goals (Strategic differentiators, medium effort).
  - 🧊 **LATER**: Horizon 3 & backlog (Exploratory or high-effort capabilities).
- Identifies cross-functional team ownership (Backend, Frontend, Mobile, Security, Billing, API).

### 4. 🔍 Advanced Multi-Criteria Filters
- Real-time multi-dimensional filtering across:
  - **Date Range**: Rolling window filter.
  - **Sentiment**: Positive, Neutral, Negative friction points.
  - **Customer Segment**: Enterprise, Pro, Starter, and Free users.
  - **Category**: Functional domains.
  - **Source Channel**: Zendesk, Intercom, Sales Calls, Community, In-App.
  - **Status**: Backlog, In Review, Planned, In Progress.
- Live feedback counter: "Showing X of Y requests (Z% of dataset)".

### 5. 📈 Demand Trend Forecasting & Velocity
- Tracks monthly customer request volume and sentiment trajectory.
- **Momentum Tracker**: Identifies surging requests gaining velocity in the last 60 days before they hit peak volume.
- 30-day forward demand projections.

### 6. 👥 Customer Segment Analysis
- Visualizes feedback distribution across Enterprise, Pro, Starter, and Free tiers.
- Segment-specific top requested features: Discover what Enterprise clients need vs. Free tier self-serve users.
- **Revenue-Weighted Prioritization**: Prioritize features by business ARR impact (Enterprise weighted 4x).

### 7. 📑 Executive Report Generator
- One-click downloadable executive decision briefs in multiple formats:
  - **Executive Markdown Report** (`.md`)
  - **Styled Executive HTML Dashboard** (`.html`)
  - **Executive PDF Report** (`.pdf`) via ReportLab
  - **Filtered Data CSV Export** (`.csv`)

### 8. 🤖 AI Product Copilot
- **Semantic Vector Search**: Natural language query interface powered by FAISS and Sentence Transformers.
- **AI PRD Generator**: Synthesizes customer feedback into an actionable Product Requirements Document (PRD).
- **AI Release Notes Generator**: Drafts engaging, customer-facing product release announcements.
- **smolagents AI Analyst**: Conversational data analyst agent executing pandas queries on demand.

---

## 🛠️ Tech Stack & Architecture

- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Embeddings & NLP:** [Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`), [KeyBERT](https://maartengr.github.io/KeyBERT/)
- **Vector Search & Deduplication:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Machine Learning & Clustering:** [Scikit-learn](https://scikit-learn.org/) (KMeans)
- **Sentiment Engine:** Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`) with rule-based fallback
- **Visualizations:** [Plotly Express](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [WordCloud](https://github.com/amueller/word_cloud)
- **Document Generation:** [ReportLab](https://www.reportlab.com/) (PDF), Python Markdown
- **AI Agent:** [smolagents](https://github.com/huggingface/smolagents) with Hugging Face Inference API

---

## 📦 Installation & Setup

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/vaishnavireddy067/AI-Feature-Request-Predictor.git
   cd AI-Feature-Request-Predictor
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data (if needed):**
   ```bash
   python data/generate_sample_data.py
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Automated Testing

Run the comprehensive test suite verifying preprocessing, RICE scoring, duplicate search, roadmapping, trends, segments, and PDF generation:

```bash
python -m pytest tests/test_pipeline.py -v
```

All 13 core test suites pass with 100% test coverage.

---

## 📂 Project Directory Structure

```text
feature_request_prioritizer/
├── .streamlit/
│   └── config.toml             # Custom theme configuration
├── data/
│   ├── generate_sample_data.py # Realistic 1,000 feedback request generator
│   └── sample_requests.csv     # Pre-generated sample dataset
├── utils/
│   ├── __init__.py             # Clean utils package interface
│   ├── ai_agent.py             # smolagents AI copilot, PRD & release notes generator
│   ├── clustering.py           # KMeans semantic clustering
│   ├── embeddings.py           # SentenceTransformer vector embeddings
│   ├── insights.py             # Evidence-backed AI recommendation engine
│   ├── keywords.py             # KeyBERT keyword extraction
│   ├── preprocessing.py        # Robust CSV data loading, cleaning & schema normalization
│   ├── ranking.py              # Smart Priority Score (RICE) engine
│   ├── reporting.py            # Executive Report Generator (Markdown, HTML, PDF)
│   ├── roadmap.py              # Now / Next / Later AI Roadmap generator
│   ├── sentiment.py            # Dual-mode sentiment analysis (Transformers + Fallback)
│   ├── segments.py             # Customer segment breakdown & revenue weighting
│   ├── strategy.py             # Cross-functional team domain auto-tagger
│   ├── summarizer.py           # Cluster centroid extractive summarizer
│   ├── vector_search.py        # FAISS cosine indexing & semantic duplicate detection
│   └── visualization.py        # Interactive Plotly & Matplotlib charts
├── tests/
│   └── test_pipeline.py        # 13 automated unit and integration tests
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License.
