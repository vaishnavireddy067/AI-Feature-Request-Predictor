import pytest
import pandas as pd
import numpy as np
import io
import os
import sys

# Add parent directory to sys.path so utils can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_and_clean_data
from utils.ranking import calculate_rice_score, rank_features, estimate_theme_effort
from utils.vector_search import build_faiss_index, find_duplicates, semantic_search
from utils.clustering import perform_clustering
from utils.roadmap import generate_ai_roadmap
from utils.trends import analyze_demand_trends
from utils.segments import analyze_customer_segments
from utils.reporting import generate_markdown_report, generate_html_report, generate_pdf_report
from utils.ai_agent import generate_prd, generate_release_notes
from utils.strategy import tag_dataframe, auto_tag_request

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "Request_ID": [f"REQ-{i:03d}" for i in range(1, 21)],
        "Date": ["2026-11-10", "2026-11-12", "2026-10-01", "2026-08-15"] * 5,
        "Category": ["Security", "Payments", "Dashboard", "API", "Mobile App"] * 4,
        "Customer_Segment": ["Enterprise", "Pro", "Free", "Starter"] * 5,
        "Source": ["Zendesk", "Intercom", "Sales Call", "In-App Feedback"] * 5,
        "Status": ["Backlog", "In Review", "Planned", "In Progress"] * 5,
        "Request_Text": [
            "We need Two-Factor Authentication via Authenticator app.",
            "Stripe payment integration is failing on checkout.",
            "Please add dark mode toggle to the analytics dashboard.",
            "GraphQL endpoint documentation is missing code examples.",
            "Mobile app crashes immediately on iPad startup."
        ] * 4,
        "Upvotes": [50, 30, 20, 10, 5] * 4
    })

# ----------------- 1. PREPROCESSING TESTS ----------------- #
def test_preprocessing_standard(sample_raw_df):
    csv_buf = io.StringIO(sample_raw_df.to_csv(index=False))
    cleaned = load_and_clean_data(csv_buf)
    assert len(cleaned) == 20
    assert "Cleaned_Text" in cleaned.columns
    assert "Category" in cleaned.columns
    assert "Upvotes" in cleaned.columns
    assert "Customer_Segment" in cleaned.columns
    assert cleaned['Upvotes'].dtype == int

def test_preprocessing_non_standard_columns():
    raw_csv = """feedback,user_tier,likes
We want PDF invoices,Enterprise,42
Faster search bar,Free,12
"""
    cleaned = load_and_clean_data(io.StringIO(raw_csv))
    assert len(cleaned) == 2
    assert "Request_Text" in cleaned.columns
    assert cleaned.iloc[0]['Request_Text'] == "We want PDF invoices"
    assert cleaned.iloc[0]['Customer_Segment'] == "Enterprise"
    assert cleaned.iloc[0]['Upvotes'] == 42
    assert "Date" in cleaned.columns
    assert "Status" in cleaned.columns

def test_preprocessing_empty_fails():
    with pytest.raises(Exception):
        load_and_clean_data(io.StringIO(""))

# ----------------- 2. RICE RANKING TESTS ----------------- #
def test_rice_calculation():
    # Reach=1000, Impact=2.0, Confidence=0.8, Effort=4.0 -> (1000 * 2 * 0.8) / 4 = 400.0
    score = calculate_rice_score(1000, 2.0, 0.8, 4.0)
    assert score == 400.0

def test_rice_zero_effort_safety():
    # Effort should be clamped to min 0.5 to prevent division by zero
    score = calculate_rice_score(100, 1.0, 1.0, 0.0)
    assert score == 200.0

def test_rank_features(sample_raw_df):
    sample_raw_df['Cluster'] = [0, 0, 1, 1, 2] * 4
    sample_raw_df['Sentiment'] = ['NEGATIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'POSITIVE'] * 4
    
    ranking = rank_features(sample_raw_df)
    assert len(ranking) == 3
    assert 'RICE Score' in ranking.columns
    assert 'Reach' in ranking.columns
    assert 'Impact' in ranking.columns
    assert 'Effort' in ranking.columns
    assert ranking['RICE Score'].iloc[0] >= ranking['RICE Score'].iloc[-1]

# ----------------- 3. VECTOR SEARCH & DEDUPLICATION TESTS ----------------- #
def test_faiss_and_deduplication():
    # Create 4 synthetic vectors: vector 0 and vector 1 are identical (distance 0 / sim 1.0)
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype='float32')
    
    index = build_faiss_index(embeddings)
    assert index.ntotal == 4
    
    duplicates = find_duplicates(index, embeddings, similarity_threshold=0.9)
    assert len(duplicates) == 1
    assert duplicates[0]['count'] == 2
    assert duplicates[0]['primary_idx'] == 0
    assert 1 in duplicates[0]['duplicate_indices']

def test_faiss_search():
    embeddings = np.eye(3, dtype='float32')
    index = build_faiss_index(embeddings)
    query = np.array([1.0, 0.0, 0.0], dtype='float32')
    sims, indices = semantic_search(query, index, k=2)
    assert len(indices) == 2
    assert indices[0] == 0
    assert pytest.approx(sims[0], 0.01) == 1.0

# ----------------- 4. ROADMAP GENERATION TESTS ----------------- #
def test_ai_roadmap_generation(sample_raw_df):
    sample_raw_df['Cluster'] = [0, 0, 1, 1, 2] * 4
    sample_raw_df['Sentiment'] = ['NEGATIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'POSITIVE'] * 4
    sample_raw_df = tag_dataframe(sample_raw_df, text_col='Request_Text')
    
    ranking = rank_features(sample_raw_df)
    keywords = {0: ["Security", "2fa"], 1: ["Dashboard", "Dark"], 2: ["Mobile", "Crash"]}
    summaries = {0: "Two-Factor Auth needed", 1: "Dark mode", 2: "Mobile crashes"}
    
    roadmap = generate_ai_roadmap(sample_raw_df, ranking, keywords, summaries)
    assert len(roadmap) == 3
    assert set(roadmap['Stage'].unique()).issubset({"NOW", "NEXT", "LATER"})
    assert 'Assigned Team' in roadmap.columns

# ----------------- 5. TREND FORECASTING TESTS ----------------- #
def test_demand_trends(sample_raw_df):
    sample_raw_df['Cluster'] = [0, 0, 1, 1, 2] * 4
    ranking = rank_features(sample_raw_df)
    keywords = {0: ["Security"], 1: ["Dashboard"], 2: ["Mobile"]}
    
    trends_df, monthly_vol = analyze_demand_trends(sample_raw_df, ranking, keywords)
    assert not trends_df.empty
    assert "Momentum" in trends_df.columns
    assert "Growth_Rate_Pct" in trends_df.columns
    assert "Forecast_30d_Requests" in trends_df.columns

# ----------------- 6. CUSTOMER SEGMENTS TESTS ----------------- #
def test_customer_segments(sample_raw_df):
    sample_raw_df['Cluster'] = [0, 0, 1, 1, 2] * 4
    ranking = rank_features(sample_raw_df)
    keywords = {0: ["Security"], 1: ["Dashboard"], 2: ["Mobile"]}
    
    top_per_seg, seg_summary, rev_df = analyze_customer_segments(sample_raw_df, ranking, keywords)
    assert not seg_summary.empty
    assert "Enterprise" in top_per_seg
    assert not rev_df.empty
    assert "Revenue_Weighted_Score" in rev_df.columns

# ----------------- 7. REPORTING GENERATION TESTS ----------------- #
def test_report_generation(sample_raw_df):
    sample_raw_df['Cluster'] = [0, 0, 1, 1, 2] * 4
    ranking = rank_features(sample_raw_df)
    keywords = {0: ["Security"], 1: ["Dashboard"], 2: ["Mobile"]}
    summaries = {0: "Two-factor auth", 1: "Dashboard layout", 2: "Mobile crash"}
    roadmap = generate_ai_roadmap(sample_raw_df, ranking, keywords, summaries)
    
    insights = [{
        "Cluster_ID": 0, "Theme": "Security", "RICE_Score": 120.0,
        "Request_Count": 8, "Pct_Of_Total": 40.0, "Total_Upvotes": 200,
        "Rationale_Badge": "Critical Blocker", "AI_Rationale": "High enterprise churn risk",
        "Actionable_Recommendation": "Sprint Inclusion", "Segment_Summary": "Enterprise: 8 (100%)",
        "Customer_Quotes": [{"text": "We need 2FA immediately.", "segment": "Enterprise", "source": "Zendesk", "upvotes": 50, "date": "2026-11-10"}]
    }]
    
    # 1. Markdown
    md = generate_markdown_report(sample_raw_df, ranking, roadmap, insights)
    assert "# 🚀 Executive Product Strategy" in md
    assert "RICE Score" in md
    
    # 2. HTML
    html = generate_html_report(sample_raw_df, ranking, roadmap, insights)
    assert "<!DOCTYPE html>" in html
    assert "Executive Product Strategy" in html
    
    # 3. PDF
    pdf_buf = generate_pdf_report(sample_raw_df, ranking, roadmap, insights)
    pdf_bytes = pdf_buf.getvalue()
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")

# ----------------- 8. AI COPILOT FALLBACK TESTS ----------------- #
def test_ai_copilot_templates():
    prd = generate_prd(
        hf_token="", 
        theme_name="Authentication & SSO", 
        summary="Users need Google & Okta SSO", 
        keywords="sso, auth, saml", 
        sample_requests="We need Okta SSO | SAML integration"
    )
    assert "# 📄 Product Requirements Document (PRD)" in prd
    assert "Authentication & SSO" in prd
    
    rn = generate_release_notes(
        hf_token="", 
        theme_name="Dark Mode Dashboard", 
        priority="NOW", 
        effort="3.0"
    )
    assert "Introducing Dark Mode Dashboard" in rn
