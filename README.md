# 🚀 AI Feature Request Prioritizer

An enterprise-grade, AI-powered product discovery and decision-making system ("Feedback → Decision → Execution") built with Python and Streamlit. It automates customer feedback analysis, detects duplicate requests using dense semantic vector search, clusters related feedback into cohesive strategic themes, models multi-scenario prioritization trade-offs, auto-generates Agile PRD specifications with acceptance criteria, plans team sprint capacity with overbooking detection, and maintains an explainable decision audit trail.

---

## ⚡ The Product Decision System: Feedback → Decision → Execution

Transforming unstructured customer signals into strategic, high-conviction product roadmaps:
1. **💬 Customer Feedback:** Ingest thousands of multi-channel requests & user pain points (Zendesk, Intercom, Jira).
2. **🧠 AI Discovery:** Group signals into dense semantic clusters with sentiment friction and demand trends.
3. **🎯 Priority Scenario Simulator:** Model strategic trade-offs (e.g. *Customer First / Churn Prevention* vs *Revenue First / Enterprise Growth*).
4. **🔎 Customer Evidence:** Audit verbatim customer quotes, user tiers, and source channels for 100% transparency.
5. **📝 AI Product Brief (PRD):** Auto-generate engineering-ready PRDs with Agile User Stories and BDD Acceptance Criteria.
6. **⚡ Realistic Roadmap & Capacity:** Plan sprint capacity (velocity points), detect overbooked sprints, and auto-balance.
7. **📜 Decision History & Audit Trail:** Maintain defensible governance recording why features were approved, deferred, or rejected.

---

## 🌟 Flagship Differentiator Modules

### 1. 🎯 Priority Scenario Simulator
- **Live Sensitivity Modeler:** Adjust Customer Impact ($W_I$), Development Effort penalty ($W_E$), Reach ($W_R$), Statistical Confidence ($W_C$), and Enterprise ARR Boost multipliers in real-time.
- **Scenario Comparison:** Compare two strategic frameworks side-by-side (e.g. **Customer First** vs **Revenue First** vs **Quick Wins**).
- **Rank Shift Deltas:** Instant rank delta tracking (`▲ +4`, `▼ -3`, `— 0`) and top movers analysis.
- **Visual Analytics:** Interactive horizontal Plotly grouped bar chart comparing normalized scores across scenarios.
- **One-Click Commitment:** "Apply Scenario to Roadmap" instantly updates live execution horizons.

### 2. 📝 AI Feature-to-PRD Generator
- **Comprehensive Product Specifications:** Turns any prioritized feature cluster into an engineering-ready PRD.
- **Problem Statement & Customer Friction:** Contextualized with real numbers (request volume, upvotes, negative sentiment %).
- **Target Personas:** Enterprise Admin, Daily Operator, Platform Integrator.
- **Agile User Stories:** Formal `As a [persona], I want [capability], So that [business value]` with priorities P0/P1/P2.
- **BDD Acceptance Criteria:** `Given / When / Then` scenarios covering Happy Path, Input Validation & Errors, and p95 Latency SLAs.
- **Proposed Architecture & Schema:** Frontend components, backend microservice endpoints, caching, and RBAC security.
- **Success Metrics (KPIs):** North Star Metric, 30-day Adoption Rate, Support Ticket Deflection %, and Uptime SLA.
- **Export Ready:** One-click download as Markdown (`.md`) or copyable code box for Jira and Linear.

### 3. 🔎 Customer Evidence Drill-Down
- **Transparent AI Recommendations:** Click any feature cluster to audit verbatim customer feedback, quotes, user segments, and source channels.
- **Multi-Dimensional Filters:** Filter by sentiment (Negative Friction, Positive Praise, Neutral), user segment (Enterprise, Pro, Free), and source channel (Zendesk, Intercom, Jira, Community).
- **Quote Feed:** Individual customer quote cards displaying exact quotes, customer tier badges, upvote counts, and timestamps.
- **Raw Data Export:** View full tabular data and export cluster-specific CSV files.

### 4. ⚡ Roadmap Capacity & Velocity Planner
- **Sprint Capacity Modeling:** Inputs for Engineering Team Size, Sprint Duration (weeks), and Velocity per Engineer (story points).
- **Effort-to-Points Conversion:** Standardizes domain effort into story point requirements.
- **Overbooked Sprint Detection:** Automatically detects sprints that exceed 100% capacity limit (highlighted in alert red).
- **Dependency Awareness:** Identifies execution order conflicts where a feature is scheduled before its architectural prerequisite.
- **Auto-Balance Capacity:** One-click greedy algorithm rebalances overflow items across Now, Next, and Later horizons so no sprint is overbooked.

### 5. 📜 Decision History & Explainability Audit Trail
- **Defensible Product Governance:** Logs why a feature was Approved (Prioritized), Planned (Target Q2), Deferred, or Rejected.
- **Structured Audit Fields:** Records Decision ID, Timestamp, Feature Theme, Status, PM Name, RICE score, customer requests, and rationale notes.
- **Strategic Decision Explainer:** Compare any two features side-by-side to generate an objective, data-backed comparative defense for leadership.
- **Export Audit Trail:** One-click download as Markdown (`.md`) or CSV.

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
