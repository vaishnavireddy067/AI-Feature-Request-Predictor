# 🚀 AI Feature Request Prioritizer

An enterprise-grade, AI-powered product discovery and roadmap intelligence web application built with Python and Streamlit. It automates customer feedback analysis, detects duplicate requests using dense semantic vector search, clusters related feedback into cohesive strategic themes, calculates explainable **Smart Priority Scores (RICE)** with custom weights, maps architectural dependencies and blockers, forecasts demand velocity momentum, breaks down customer segments, and auto-generates executive reports and roadmaps.

---

## 🌟 Upgrades by Sprint

### Sprint 1 — Decision Intelligence
- **1. 🎯 Smart Priority Score (RICE Framework):**
  Evaluates every feature request cluster across **Reach**, **Impact**, **Confidence**, and **Effort**:
  $$\text{Weighted RICE Score} = \frac{(\text{Reach} \times W_R) \times (\text{Impact} \times W_I) \times (\text{Confidence} \times W_C)}{\text{Effort} \times W_E}$$
  - **Reach:** User exposure weighted by request volume, upvotes, and customer tiers.
  - **Impact:** Quantified from customer sentiment friction (severe pain points) and enterprise revenue exposure.
  - **Confidence:** Statistical confidence based on feedback density and sample size.
  - **Effort:** Domain-based engineering complexity (1-10 scale), with real-time interactive sliders.
- **2. 💡 Evidence-Backed AI Insights:**
  - Real customer feedback metrics (exact request counts, % share of total, upvote velocity).
  - Customer sentiment distribution (% negative friction vs. positive).
  - **Verbatim Customer Quotes:** Real customer quotes per theme with user tier and source channel metadata.
- **3. ⚖️ Custom Scoring Weights & Strategic Presets:**
  - Sliders for $W_{\text{Reach}}$, $W_{\text{Impact}}$, $W_{\text{Confidence}}$, and $W_{\text{Effort}}$.
  - Built-in presets: *Balanced (Standard RICE)*, *Customer Friction & Churn First*, *Quick Wins & High ROI*, and *Enterprise ARR Expansion*.

---

### Sprint 2 — Product Planning
- **4. 🗺️ Interactive Roadmap Builder (Now / Next / Later):**
  - **Dynamic Stage Movement:** Reorder and move feature items between **🔥 NOW**, **⭐ NEXT**, and **🧊 LATER** with interactive dropdown selectors.
  - Cross-functional team ownership tags (Backend, Frontend, Mobile, Security, Billing, API).
  - One-click **Export Roadmap** to Markdown (`.md`) and CSV (`.csv`).
- **5. 🔗 Dependency Mapping & Blocker Detection:**
  - Identifies architectural prerequisites (e.g. *Security requires Authentication*; *Payments requires PCI Tokenization*).
  - **Automated Blocker Warnings:** Detects execution order violations (e.g. if a feature in `NOW` depends on a prerequisite in `LATER`).
  - Full dependency table showing prerequisite chains and downstream enablers.
- **6. 📑 Executive Report Generator:**
  - Downloadable executive decision briefs:
    - **Executive Markdown Report** (`.md`)
    - **Styled Executive HTML Dashboard** (`.html`)
    - **Executive PDF Report** (`.pdf`) via ReportLab
    - **Filtered Data CSV Export** (`.csv`)

---

### Sprint 3 — Analytics
- **7. 📈 Demand Trend Forecasting & Velocity:**
  - Tracks monthly customer request volume and sentiment trajectory.
  - **Momentum Tracker:** Identifies surging requests gaining velocity in the last 60 days before they hit peak volume.
  - 30-day forward demand projections.
- **8. 👥 Customer Segment Analysis:**
  - Visualizes feedback distribution across Enterprise, Pro, Starter, and Free tiers.
  - Segment-specific top requested features: Discover what Enterprise clients need vs. Free tier self-serve users.
  - **Revenue-Weighted Prioritization:** Calculates business ARR impact with Enterprise requests weighted 4x.
- **9. 🚨 Emerging Feature Alerts:**
  - Automatically flags critical signals:
    - 🚀 **Velocity Surge:** Sudden spikes in demand velocity (+45% growth).
    - 🚨 **Friction Spike:** Severe customer frustration and churn risks (>35% negative sentiment).
    - 💼 **Enterprise Breakout:** Features disproportionately requested by Enterprise accounts.
    - ⚡ **Sleeper Hits:** Niche features showing steep upward momentum.

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

Run the comprehensive test suite verifying preprocessing, RICE scoring, duplicate search, roadmapping, dependencies, alerts, trends, segments, and PDF generation:

```bash
python -m pytest tests/test_pipeline.py -v
```

All 16 core test suites pass with 100% test coverage.

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
│   ├── alerts.py               # Emerging feature alerts (Velocity, Friction, Enterprise)
│   ├── clustering.py           # KMeans semantic clustering
│   ├── dependencies.py         # Architectural dependency mapping & blocker detector
│   ├── embeddings.py           # SentenceTransformer vector embeddings
│   ├── insights.py             # Evidence-backed AI recommendation engine
│   ├── keywords.py             # KeyBERT keyword extraction
│   ├── preprocessing.py        # Robust CSV data loading, cleaning & schema normalization
│   ├── ranking.py              # Smart Priority Score (RICE) engine with custom weights
│   ├── reporting.py            # Executive Report Generator (Markdown, HTML, PDF)
│   ├── roadmap.py              # Now / Next / Later AI Roadmap generator & interactive mover
│   ├── sentiment.py            # Dual-mode sentiment analysis (Transformers + Fallback)
│   ├── segments.py             # Customer segment breakdown & revenue weighting
│   ├── strategy.py             # Cross-functional team domain auto-tagger
│   ├── summarizer.py           # Cluster centroid extractive summarizer
│   ├── trends.py               # Demand velocity & momentum trend forecasting engine
│   ├── vector_search.py        # FAISS cosine indexing & semantic duplicate detection
│   └── visualization.py        # Interactive Plotly & Matplotlib charts
├── tests/
│   └── test_pipeline.py        # 16 automated unit and integration tests
├── app.py                      # Main Streamlit application
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License.
