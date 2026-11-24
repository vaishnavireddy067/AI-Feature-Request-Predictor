# 🚀 AI Feature Request Prioritizer

An enterprise-grade, AI-assisted product discovery and decision-making system ("Feedback → Decision → Execution") built with Python and Streamlit. It automates customer feedback analysis, identifies potential semantic duplicates for human review, clusters related feedback into cohesive strategic themes, models multi-scenario prioritization trade-offs, auto-generates Agile PRD specifications with acceptance criteria, plans team sprint capacity with overbooking detection, and maintains an explainable decision audit trail.

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

## 📊 Standard Benchmark Dataset Metrics

The preloaded demo environment contains a calibrated benchmark dataset reflecting real-world enterprise SaaS feedback distributions. For any newly uploaded CSV, all metrics update dynamically:

| Metric | Benchmark Value | Description & Definition |
| :--- | :--- | :--- |
| **Total Feedback** | **1,248** | Total multi-channel customer requests across Enterprise, Pro, Starter, and Free tiers |
| **Feature Clusters** | **18** | Distinct functional themes discovered through semantic embedding and clustering |
| **High Priority Features** | **5** | Core initiatives scheduled in the **NOW** execution horizon (Smart RICE Score > 7.5) |
| **Emerging Trends** | **3** | Active alerts flagged for demand acceleration (+45% to +60% velocity surge in the last 30 days) |
| **Sentiment Distribution** | **72% Pos / 16% Neu / 12% Neg** | Calibrated customer sentiment trajectory (898 Positive, 199 Neutral, 151 Friction points) |
| **Sprint Feasibility** | **100% (5/5 unblocked)** | Proportion of features in the **NOW** horizon with all architectural prerequisites validated |

---

## 📐 Scoring Framework: The Weighted RICE Formula

To balance objectivity with strategic flexibility, features are evaluated using the **Weighted RICE Framework**. When all weights are set to $1.0$, the formula simplifies identically to standard RICE.

$$\text{Smart RICE} = \frac{(\text{Reach} \times W_R) \times (\text{Impact} \times W_I) \times (\text{Confidence} \times W_C)}{\max(0.5, \text{Effort} \times W_E)}$$

$$\text{Standard RICE (Baseline when all } W = 1.0\text{)} = \frac{\text{Reach} \times \text{Impact} \times \text{Confidence}}{\text{Effort}}$$

### Input Scales & Definitions:
1. **Reach ($R$) — Affected User Volume**
   - *Scale:* $50$ to $1,500$ monthly affected users.
   - *Calculation:* $\text{Reach} = (N_{\text{requests}} \times 10) + (N_{\text{upvotes}} \times 2) + (N_{\text{Enterprise}} \times 20) + (N_{\text{Pro}} \times 5)$
   - *$W_R$ (Reach Weight, $0.2 - 2.5$):* Adjusts sensitivity to broad volume vs high-value accounts.
2. **Impact ($I$) — Business & Retention Impact**
   - *Scale:* Qualitative multiplier ($0.5$ to $3.0$):
     - `3.0` (Massive): Critical Enterprise blocker with $>40\%$ negative sentiment.
     - `2.0` (High): Significant customer friction or major revenue driver.
     - `1.5` (Medium-High): Broad user demand across $>8\%$ of total feedback.
     - `1.0` (Medium): Standard expected functional improvement.
     - `0.5` (Low): Minor cosmetic polish or primarily free-tier request.
   - *$W_I$ (Impact Weight, $0.2 - 2.5$):* Emphasizes customer friction to combat churn.
3. **Confidence ($C$) — Statistical Sample Confidence**
   - *Scale:* Evidence confidence factor ($0.50$ to $1.00$):
     - `1.0` (100%): Very high sample density ($N \ge 40$ requests).
     - `0.9` (90%): Substantial sample size ($N \ge 20$ requests).
     - `0.8` (80%): Moderate evidence ($N \ge 10$ requests).
     - `0.65` (65%): Early signal ($N \ge 5$ requests).
     - `0.5` (50%): Low sample size ($N < 5$ requests).
   - *$W_C$ (Confidence Weight, $0.2 - 2.5$):* Penalizes speculative or low-sample requests.
4. **Effort ($E$) — Engineering Complexity**
   - *Scale:* Story point complexity units ($1.0$ to $10.0$ points):
     - Estimated by architectural domain: UI/UX (3.0), Notifications (3.5), Dashboard (4.0), Reports (4.5), Performance (5.5), Mobile (6.0), API (6.5), Payments (7.5), Security/Auth (8.0). Clamped to min 0.5.
   - *$W_E$ (Effort Penalty, $0.2 - 2.5$):* Multiplies effort penalty. Higher $W_E$ heavily favors Quick Wins.

---

## 🌟 Flagship Differentiator Modules

### 1. 🎯 Priority Scenario Simulator
- **Dynamic Sensitivity Modeler:** Adjust Customer Impact ($W_I$), Development Effort penalty ($W_E$), Reach ($W_R$), Statistical Confidence ($W_C$), and Enterprise ARR Boost multipliers dynamically.
- **Scenario Comparison:** Compare two strategic frameworks side-by-side (e.g. **Customer First** vs **Revenue First** vs **Quick Wins**).
- **Rank Shift Deltas:** Dynamic rank delta tracking (`▲ +4`, `▼ -3`, `— 0`) and top movers analysis.
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

## 🛠️ Tech Stack & Architecture

- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Embeddings & NLP:** [Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`), [KeyBERT](https://maartengr.github.io/KeyBERT/)
- **Vector Search & Deduplication:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Machine Learning & Clustering:** [Scikit-learn](https://scikit-learn.org/) (KMeans)
- **Sentiment Engine:** Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`) with rule-based fallback
- **Visualizations:** [Plotly Express](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/)
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

Run the automated test suite verifying preprocessing, RICE scoring, vector deduplication, roadmapping, dependencies, alerts, trends, segments, scenario simulation, capacity planning, and PRD generation:

```bash
python -m pytest tests/test_pipeline.py -v
```

**Test Status:** 20 test cases passing across data preprocessing, RICE scoring, vector deduplication, scenario simulation, capacity planning, and PRD generation.

---

## 📂 Project Directory Structure

```text
feature_request_prioritizer/
├── .streamlit/
│   └── config.toml             # Custom theme configuration
├── data/
│   ├── generate_sample_data.py # Calibrated 1,248 feedback request generator across 18 clusters
│   ├── sample_processed.pkl    # Precomputed benchmark dataset cache
│   └── sample_requests.csv     # 1,248 sample customer feedback requests
├── utils/
│   ├── __init__.py             # Clean utils package interface
│   ├── ai_agent.py             # smolagents AI copilot & release notes generator
│   ├── alerts.py               # Automatically detected feature alerts (Velocity, Friction, Enterprise)
│   ├── capacity.py             # Roadmap capacity planner & auto-balancing engine
│   ├── clustering.py           # KMeans semantic clustering
│   ├── decision_log.py         # Decision history logger & comparative defense explainer
│   ├── dependencies.py         # Architectural dependency mapping & blocker detector
│   ├── embeddings.py           # SentenceTransformer vector embeddings
│   ├── insights.py             # Evidence-backed AI recommendation engine
│   ├── keywords.py             # KeyBERT keyword extraction
│   ├── prd_generator.py        # Comprehensive AI Feature-to-PRD generator
│   ├── preprocessing.py        # Robust CSV data loading, cleaning & schema normalization
│   ├── ranking.py              # Smart Priority Score (Weighted RICE) engine
│   ├── reporting.py            # Executive Report Generator (Markdown, HTML, PDF)
│   ├── roadmap.py              # Now / Next / Later AI Roadmap generator & interactive mover
│   ├── sentiment.py            # Dual-mode sentiment analysis (Transformers + Fallback)
│   ├── segments.py             # Customer segment breakdown & revenue-weighted estimation
│   ├── simulator.py            # Priority scenario simulator & side-by-side comparative modeler
│   ├── strategy.py             # Cross-functional team domain auto-tagger
│   ├── summarizer.py           # Cluster centroid extractive summarizer
│   ├── trends.py               # Demand velocity & momentum trend forecasting engine
│   ├── vector_search.py        # FAISS cosine indexing & potential semantic duplicate detection
│   └── visualization.py        # Interactive Plotly & Matplotlib charts
├── tests/
│   └── test_pipeline.py        # 20 automated unit and integration tests
├── app.py                      # Main Streamlit application
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License.
