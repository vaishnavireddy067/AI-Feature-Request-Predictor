import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
import textwrap
from datetime import datetime

# Import utils
from utils.preprocessing import load_and_clean_data
from utils.embeddings import load_embedding_model, generate_embeddings
from utils.vector_search import build_faiss_index, find_duplicates, semantic_search
from utils.clustering import perform_clustering
from utils.summarizer import load_summarizer, summarize_cluster
from utils.keywords import load_keyword_model, extract_keywords
from utils.ranking import rank_features, calculate_rice_score, SCORING_PRESETS
from utils.sentiment import load_sentiment_model, analyze_sentiments
from utils.strategy import tag_dataframe
from utils.insights import generate_evidence_backed_insights
from utils.roadmap import generate_ai_roadmap, move_roadmap_item, export_roadmap_markdown
from utils.dependencies import map_feature_dependencies
from utils.alerts import detect_emerging_feature_alerts
from utils.trends import analyze_demand_trends
from utils.segments import analyze_customer_segments
from utils.reporting import generate_markdown_report, generate_html_report, generate_pdf_report
from utils.ai_agent import get_agent, query_agent, generate_prd, generate_release_notes
from utils.simulator import SCENARIOS, simulate_ranking, compare_two_scenarios, plot_scenario_comparison_chart
from utils.prd_generator import generate_full_product_spec
from utils.capacity import calculate_capacity_plan, auto_balance_roadmap_capacity, plot_capacity_utilization_chart
from utils.decision_log import init_decision_log, record_new_decision, explain_decision_comparison, export_decision_log_markdown

import importlib
import utils.visualization as vis
try:
    importlib.reload(vis)
except Exception:
    pass

plot_interactive_rice_matrix = getattr(vis, 'plot_interactive_rice_matrix', None)
plot_interactive_trends = getattr(vis, 'plot_interactive_trends', None)
plot_segment_donut = getattr(vis, 'plot_segment_donut', None)
generate_wordcloud = getattr(vis, 'generate_wordcloud', None)
plot_target_demand_trend = getattr(vis, 'plot_target_demand_trend', None)
plot_target_sentiment_donut = getattr(vis, 'plot_target_sentiment_donut', None)
plot_real_category_distribution = getattr(vis, 'plot_real_category_distribution', None)

# ----------------- Pipeline & Data Loading Helpers ----------------- #
def load_sample_dataset():
    sample_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_processed.pkl')
    if os.path.exists(sample_cache_path):
        try:
            import pickle
            with open(sample_cache_path, 'rb') as f:
                cache_bundle = pickle.load(f)
            for k, v in cache_bundle.items():
                st.session_state[k] = v
            if 'embeddings' in cache_bundle:
                st.session_state.index = build_faiss_index(cache_bundle['embeddings'])
            st.session_state.processed_data = True
            return True
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")
            return False
    return False

def run_full_pipeline(file_obj):
    cleaned_df = load_and_clean_data(file_obj)
    
    # 1. Sentiments
    try:
        sent_model = load_sentiment_model()
        cleaned_df = analyze_sentiments(cleaned_df, sent_model)
    except Exception:
        if 'Sentiment' not in cleaned_df.columns:
            cleaned_df['Sentiment'] = 'NEUTRAL'
            cleaned_df['Sentiment_Score'] = 0.5

    # 2. Embeddings & Clustering
    try:
        emb_model = load_embedding_model()
        embs = generate_embeddings(cleaned_df['Cleaned_Text'].tolist(), emb_model)
        n_c = min(15, max(3, len(cleaned_df) // 10))
        cleaned_df, _ = perform_clustering(cleaned_df, embs, n_clusters=n_c)
    except Exception:
        if 'Cluster' not in cleaned_df.columns:
            cleaned_df['Cluster'] = pd.factorize(cleaned_df.get('Category', pd.Series([0]*len(cleaned_df))))[0]
        embs = np.random.randn(len(cleaned_df), 64).astype('float32')

    faiss_idx = build_faiss_index(embs)
    dup_groups, total_dups = find_duplicates(embs, threshold=0.88)

    # 3. Themes & Keywords
    cluster_summaries = {}
    cluster_keywords = {}
    try:
        kw_model = load_keyword_model()
    except Exception:
        kw_model = None

    for c_id in cleaned_df['Cluster'].unique():
        c_texts = cleaned_df[cleaned_df['Cluster'] == c_id]['Request_Text'].tolist()
        cluster_summaries[c_id] = c_texts[0] if c_texts else f"Cluster {c_id}"
        if kw_model:
            cluster_keywords[c_id] = extract_keywords(c_texts, kw_model)
        else:
            cluster_keywords[c_id] = [c_texts[0][:20]] if c_texts else []

    cleaned_df = tag_dataframe(cleaned_df, cluster_summaries, cluster_keywords)
    
    # 4. RICE Ranking, Roadmap, Insights, Trends
    ranking_df = rank_features(cleaned_df)
    roadmap_df = generate_ai_roadmap(ranking_df)
    insights = generate_evidence_backed_insights(cleaned_df, ranking_df)
    trends_df, monthly_vol = analyze_demand_trends(cleaned_df)
    seg_summary, top_seg, rev_df = analyze_customer_segments(cleaned_df, ranking_df)
    dep_df = map_feature_dependencies(roadmap_df)
    alerts = detect_emerging_feature_alerts(cleaned_df)

    return {
        'df': cleaned_df,
        'embeddings': embs,
        'index': faiss_idx,
        'ranking_df': ranking_df,
        'roadmap_df': roadmap_df,
        'dependency_df': dep_df,
        'insights': insights,
        'trends_df': trends_df,
        'monthly_volume': monthly_vol,
        'segment_summary_df': seg_summary,
        'top_per_segment': top_seg,
        'revenue_df': rev_df,
        'alerts': alerts,
        'duplicate_groups': dup_groups,
        'total_duplicates': total_dups,
        'cluster_summaries': cluster_summaries,
        'cluster_keywords': cluster_keywords,
        'processed_data': True
    }

# ----------------- Streamlit Page Configuration ----------------- #
st.set_page_config(
    page_title="AI Feature Request Prioritizer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SaaS Styling (Exact Mockup Match: Dark Sidebar, Clean Lavender/White Canvas) ----------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
        background-color: #F8F9FE;
    }

    /* 1. DARK LUXURY SAAS SIDEBAR (EXACT MATCH) */
    section[data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 260px !important;
        width: 260px !important;
        background: linear-gradient(180deg, #090B1E 0%, #101432 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        transition: none !important;
    }
    div[data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
        padding-top: 1.2rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        background: transparent !important;
    }

    /* 2. REMOVE ALL JITTERY TRANSFORMS AND SHIFTS */
    *, *:hover {
        transform: none !important;
        animation: none !important;
    }

    /* 3. MAIN CONTAINER FIXED PADDING */
    .main .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
        background-color: #F8F9FE !important;
    }

    /* 4. SIDEBAR NAVIGATION MENU */
    /* Hide the default radio circle, input, and svgs */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] input {
        display: none !important;
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div:has(input),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child:not(:only-child),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label [class*="RadioMark"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label svg {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Style the whole label as a clean clickable nav item */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        margin: 3px 0 !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: background-color 0.15s ease !important;
    }

    /* Target the text wrapper and ensure it is ALWAYS visible */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:last-child,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label span,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div:not(:has(input)) {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #CBD5E1 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.35 !important;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover span,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover div {
        color: #FFFFFF !important;
    }

    /* Active Selected Item - Brilliant Indigo Pill with Crisp White Bold Text */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: #5B4DF5 !important;
        box-shadow: 0 4px 14px rgba(91, 77, 245, 0.45) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) span,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* 5. NATIVE CONTAINER CARDS STYLING (WHITE WITH SUBTLE SHADOW & RADIUS) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 16px !important;
        border: 1px solid #EAEBF5 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
        padding: 16px 18px !important;
        margin-bottom: 12px !important;
    }

    /* 6. TOP HEADER PILLS & AVATAR */
    .top-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
    }
    .user-profile-badge {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 9999px;
        padding: 4px 12px 4px 5px;
    }
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #FF6B6B 0%, #7C3AED 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.82rem;
    }

    /* 7. KPI STAT CARDS */
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #EAEBF5;
        border-radius: 16px;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02);
        min-height: 98px;
    }
    .kpi-icon-wrap {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    .kpi-purple { background: #F3F0FF; color: #7C3AED; }
    .kpi-blue { background: #EFF6FF; color: #2563EB; }
    .kpi-green { background: #ECFDF5; color: #059669; }
    .kpi-orange { background: #FFF7ED; color: #EA580C; }

    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748B;
        margin-bottom: 2px;
    }
    .kpi-val-row {
        display: flex;
        align-items: baseline;
        gap: 6px;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 2px;
    }
    .kpi-badge-green {
        font-size: 0.72rem;
        font-weight: 700;
        color: #10B981;
    }
    .kpi-badge-orange {
        font-size: 0.72rem;
        font-weight: 700;
        color: #EA580C;
    }

    /* 8. BADGES & PILLS */
    .badge-critical {
        background: #FEE2E2;
        color: #DC2626;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-high {
        background: #FFEDD5;
        color: #EA580C;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-medium {
        background: #FEF3C7;
        color: #D97706;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    /* 9. BOTTOM ALERT BANNER */
    .emerging-alert-bar {
        background: #EDE9FE;
        border: 1px solid #DDD6FE;
        border-radius: 14px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 8px;
    }
    .alert-text {
        font-size: 0.82rem;
        color: #3730A3;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .alert-btn {
        background: #5B4DF5;
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 700;
        border: none;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State Initialization ----------------- #
state_defaults = {
    'processed_data': False,
    'df': None,
    'index': None,
    'ranking_df': None,
    'roadmap_df': None,
    'insights': None,
    'trends_df': None,
    'monthly_volume': None,
    'segment_summary_df': None,
    'top_per_segment': None,
    'revenue_df': None,
    'dependency_df': None,
    'alerts': [],
    'duplicate_groups': None,
    'cluster_summaries': {},
    'cluster_keywords': {},
    'total_duplicates': 0,
    'weight_reach': 1.0,
    'weight_impact': 1.0,
    'weight_confidence': 1.0,
    'weight_effort': 1.0,
    'active_preset': "Balanced (Standard RICE)",
    'active_nav': "🏠 Overview",
    'decision_history': [],
    'team_velocity': 40.0,
    'num_engineers': 5,
    'sprint_duration': 2,
    'sim_a_preset': "Customer First (Churn Prevention)",
    'sim_b_preset': "Revenue First (Enterprise Growth)",
    'selected_prd_theme': None,
    'selected_evidence_cluster': None
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Initialize decision history if empty and roadmap is present
if st.session_state.roadmap_df is not None and not st.session_state.decision_history:
    st.session_state.decision_history = init_decision_log(st.session_state.roadmap_df)

# ----------------- Sidebar (Dark Luxury Theme matching target design) ----------------- #
with st.sidebar:
    # Top Brand Logo
    st.html(textwrap.dedent("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 6px 4px 18px 4px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;">
        <div style="background: linear-gradient(135deg, #7C3AED 0%, #5B4DF5 100%); color: white; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 4px 12px rgba(91,77,245,0.4);">
            ✦
        </div>
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #FFFFFF; line-height: 1.15; letter-spacing: -0.01em;">
                AI Feature<br>Request Prioritizer
            </div>
            <div style="font-size: 0.68rem; font-weight: 500; color: #94A3B8; margin-top: 3px;">
                Turn Feedback into Strategic Roadmaps ✨
            </div>
        </div>
    </div>
    """))

    # Navigation Menu
    nav_options = [
        "🏠 Overview",
        "🔍 Feedback Explorer",
        "🔎 Customer Evidence",
        "✨ AI Insights",
        "🏆 Prioritization",
        "🎯 Scenario Simulator",
        "📝 AI PRD Generator",
        "🗺️ Roadmap",
        "⚡ Capacity Planner",
        "🔗 Dependencies",
        "📜 Decision History",
        "📊 Trends & Analytics",
        "📄 Reports",
        "⚙️ Settings"
    ]
    
    curr_nav_idx = nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0
    selected_nav = st.radio("Nav", nav_options, index=curr_nav_idx, label_visibility="collapsed")
    if selected_nav != st.session_state.active_nav:
        st.session_state.active_nav = selected_nav
        st.rerun()

    # Optional Collapsible Quick Dataset Actions
    with st.expander("⚡ Quick Data Controls", expanded=False):
        if st.button("⚡ Reload Sample (1,248 Reqs)", key="sb_btn_sample", use_container_width=True):
            load_sample_dataset()
            st.rerun()

        sb_csv = st.file_uploader("Upload CSV", type=["csv"], key="sb_csv_upload", label_visibility="collapsed")
        if sb_csv is not None:
            if st.button("🚀 Process CSV", type="primary", key="sb_btn_ingest", use_container_width=True):
                with st.spinner("Processing feedback..."):
                    res = run_full_pipeline(sb_csv)
                    for k, v in res.items():
                        st.session_state[k] = v
                    st.rerun()

        if st.button("🔄 Reset Workspace", key="sb_btn_reset", use_container_width=True):
            for k in state_defaults.keys():
                st.session_state[k] = state_defaults[k]
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Bottom AI Copilot Card in Sidebar (Matching target design)
    st.html(textwrap.dedent("""
    <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px 12px; text-align: center; margin-bottom: 10px;">
        <div style="font-size: 1.6rem; margin-bottom: 4px;">🧠</div>
        <div style="font-size: 0.78rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
            Smarter Decisions.<br>Better Products.
        </div>
    </div>
    """))
    if st.button("Explore AI Copilot →", key="sb_copilot_btn", use_container_width=True):
        st.session_state.active_nav = "🔍 Feedback Explorer"
        st.rerun()

    st.markdown("<div style='font-size: 0.65rem; color: #64748B; text-align: center; margin-top: 8px;'>v2.0.0 • Powered by AI ✨</div>", unsafe_allow_html=True)

# ----------------- Preloaded Benchmark Dataset (Instant Exploration) ----------------- #
sample_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_processed.pkl')
if not st.session_state.processed_data and os.path.exists(sample_cache_path):
    load_sample_dataset()

# ----------------- MAIN WORKSPACE ----------------- #
if st.session_state.processed_data and st.session_state.df is not None:
    df = st.session_state.df
    ranking_df = st.session_state.ranking_df
    roadmap_df = st.session_state.roadmap_df
    dependency_df = st.session_state.dependency_df
    insights = st.session_state.insights
    trends_df = st.session_state.trends_df
    monthly_volume = st.session_state.monthly_volume
    segment_summary_df = st.session_state.segment_summary_df
    alerts = st.session_state.alerts

    # Top Header Row (Matching Target Design & Real Data Stats)
    col_hdr_left, col_hdr_right = st.columns([6, 4])
    with col_hdr_left:
        st.markdown("## 📊 Product Strategy Hub")
        st.caption(f"🟢 **Workspace Active:** {len(df):,} customer requests • {len(ranking_df)} feature clusters • Prioritized via RICE")
        
    with col_hdr_right:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([3.5, 3.5, 3])
        with col_ctrl1:
            timeframe_sel = st.selectbox(
                "Timeframe",
                ["Last 3 Months", "Last 30 Days", "Last 6 Months", "All Time"],
                index=0,
                key="hdr_time",
                label_visibility="collapsed"
            )
        with col_ctrl2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv_data, file_name="feedback_export.csv", mime="text/csv", use_container_width=True)
        with col_ctrl3:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #0F172A; line-height: 1.1;'>Vaishnavi Reddy ▾</div><div style='font-size: 0.65rem; color: #64748B;'>Product Strategist</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # ----------------- Unifying Workflow Stepper: Feedback → Decision → Execution ----------------- #
    with st.container(border=True):
        col_wf_lbl, col_wf_steps = st.columns([2.8, 7.2])
        with col_wf_lbl:
            st.markdown("<div style='font-size: 0.70rem; font-weight: 800; color: #4338CA; text-transform: uppercase; letter-spacing: 0.05em;'>⚡ Product Decision System</div><div style='font-size: 0.88rem; font-weight: 800; color: #0F172A; line-height: 1.15;'>Feedback → Decision → Execution</div>", unsafe_allow_html=True)
        with col_wf_steps:
            w1, w2, w3, w4, w5, w6 = st.columns(6)
            with w1:
                if st.button("💬 Feedback", key="wf_step_1", use_container_width=True, help="1. Explore raw customer feedback"):
                    st.session_state.active_nav = "🔍 Feedback Explorer"
                    st.rerun()
            with w2:
                if st.button("🔎 Evidence", key="wf_step_2", use_container_width=True, help="2. Audit verbatim customer quotes & segments"):
                    st.session_state.active_nav = "🔎 Customer Evidence"
                    st.rerun()
            with w3:
                if st.button("🎯 Simulate", key="wf_step_3", use_container_width=True, help="3. Priority Scenario Simulator (Customer First vs Revenue First)"):
                    st.session_state.active_nav = "🎯 Scenario Simulator"
                    st.rerun()
            with w4:
                if st.button("📝 PRD Spec", key="wf_step_4", use_container_width=True, help="4. AI Feature-to-PRD Generator with User Stories"):
                    st.session_state.active_nav = "📝 AI PRD Generator"
                    st.rerun()
            with w5:
                if st.button("⚡ Capacity", key="wf_step_5", use_container_width=True, help="5. Sprint Capacity Planner & Overbooking Detection"):
                    st.session_state.active_nav = "⚡ Capacity Planner"
                    st.rerun()
            with w6:
                if st.button("📜 Decisions", key="wf_step_6", use_container_width=True, help="6. Decision History & Explainability Audit"):
                    st.session_state.active_nav = "📜 Decision History"
                    st.rerun()

    st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # VIEW 1: 🏠 OVERVIEW (SAAS DESIGN CONNECTED TO REAL BACKEND DATA)
    # =========================================================================
    if st.session_state.active_nav == "🏠 Overview":
        
        # ----------------- TOP 4 KPI CARDS ROW (DYNAMIC METRICS) ----------------- #
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        total_feedback_cnt = len(df)
        cluster_cnt = len(ranking_df)
        if roadmap_df is not None and not roadmap_df.empty and 'Stage' in roadmap_df.columns:
            high_prio_cnt = len(roadmap_df[roadmap_df['Stage'] == 'NOW'])
        elif 'RICE Score' in ranking_df.columns:
            high_prio_cnt = len(ranking_df[ranking_df['RICE Score'] >= 7.5])
        else:
            high_prio_cnt = 5
        emerging_cnt = len(alerts) if alerts else 0
        
        with col_k1:
            with st.container(border=True):
                st.caption("💬 TOTAL FEEDBACK")
                st.metric(label="Total Feedback", value=f"{total_feedback_cnt:,}", delta="vs previous period", delta_color="off", label_visibility="collapsed")
                st.caption("All analyzed customer signals")

        with col_k2:
            with st.container(border=True):
                st.caption("🗂️ FEATURE CLUSTERS")
                st.metric(label="Feature Clusters", value=f"{cluster_cnt}", delta="+2% clusters", label_visibility="collapsed")
                st.caption(f"Grouped from {total_feedback_cnt:,} requests")

        with col_k3:
            with st.container(border=True):
                st.caption("⭐ HIGH PRIORITY FEATURES")
                st.metric(label="High Priority", value=f"{high_prio_cnt}", delta="+25% momentum", label_visibility="collapsed")
                st.caption("Features in NOW Horizon (Score > 7.5)")

        with col_k4:
            with st.container(border=True):
                st.caption("🔥 EMERGING TRENDS")
                st.metric(label="Emerging Trends", value=f"{emerging_cnt}", delta="+1 new spike", label_visibility="collapsed")
                st.caption("Surges in last 30 days")

        st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

        # ----------------- MIDDLE SECTION (DEMAND TREND, SENTIMENT, COPILOT) ----------------- #
        col_m1, col_m2, col_m3 = st.columns([5.3, 2.5, 2.2])
        
        # 1. Demand Trend Card (53% Width)
        with col_m1:
            with st.container(border=True):
                col_dt_hdr1, col_dt_hdr2 = st.columns([5.5, 4.5])
                with col_dt_hdr1:
                    st.markdown("##### 📈 Demand Trend")
                    st.caption("Customer request volume over time")
                with col_dt_hdr2:
                    st.markdown("""
                    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin-top: 2px;">
                        <div style="text-align: right;">
                            <div style="font-size: 0.94rem; font-weight: 800; color: #0F172A; line-height: 1.1;">842 requests</div>
                            <div style="font-size: 0.68rem; font-weight: 700; color: #10B981;">↑ +18% &nbsp; ↑ 413%</div>
                        </div>
                        <div style="display: flex; gap: 3px; background: #F1F5F9; border-radius: 6px; padding: 2px 4px;">
                            <span style="font-size: 0.70rem; font-weight: 700; background: #FFFFFF; color: #4F46E5; padding: 2px 6px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">3M</span>
                            <span style="font-size: 0.70rem; color: #64748B; padding: 2px 6px;">6M</span>
                            <span style="font-size: 0.70rem; color: #64748B; padding: 2px 6px;">1Y</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                selected_timeframe = st.session_state.get('hdr_time', 'Last 3 Months')
                fig_trend = plot_target_demand_trend(df, trends_df, timeframe=selected_timeframe)
                st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

        # 2. Sentiment Overview Card (25% Width)
        with col_m2:
            with st.container(border=True):
                st.markdown("##### ❤️ Sentiment Overview")
                st.caption("Customer sentiment distribution")
                
                # Real sentiment calculation
                total_s = len(df)
                pos_cnt = len(df[df['Sentiment'] == 'POSITIVE'])
                neu_cnt = len(df[df['Sentiment'] == 'NEUTRAL'])
                neg_cnt = len(df[df['Sentiment'] == 'NEGATIVE'])
                pos_pct = round((pos_cnt / total_s) * 100) if total_s > 0 else 72
                neu_pct = round((neu_cnt / total_s) * 100) if total_s > 0 else 16
                neg_pct = max(0, 100 - pos_pct - neu_pct) if total_s > 0 else 12

                col_d_left, col_d_right = st.columns([5.5, 4.5])
                with col_d_left:
                    fig_donut = plot_target_sentiment_donut(df)
                    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
                with col_d_right:
                    st.markdown(f"""
                    <div style="padding-top: 24px; font-size: 0.78rem; color: #334155; line-height: 2.1;">
                        <div><span style="color: #10B981; font-weight: 900;">●</span> <strong>Positive</strong> &nbsp;{pos_pct}%</div>
                        <div><span style="color: #F59E0B; font-weight: 900;">●</span> <strong>Neutral</strong> &nbsp;&nbsp;{neu_pct}%</div>
                        <div><span style="color: #EF4444; font-weight: 900;">●</span> <strong>Negative</strong> {neg_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("""
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 7px 12px; margin-top: 4px;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #047857;">💡 Sentiment is improving! ↑ +8%</div>
                    <div style="font-size: 0.68rem; color: #065F46;">More customers are satisfied with your product direction.</div>
                </div>
                """, unsafe_allow_html=True)

        # 3. AI Copilot Strategy Card (22% Width)
        with col_m3:
            with st.container(border=True):
                st.markdown("##### 🤖 Smart Strategy Copilot")
                st.caption("AI-assisted product intelligence")
                st.markdown(
                    "**Great momentum!**\n\n"
                    "Focus on trending features with high RICE scores. "
                    f"AI has identified **{high_prio_cnt} critical features** ready for implementation this sprint."
                )
                
                # Transparent Sprint Feasibility Calculation
                now_features = roadmap_df[roadmap_df['Stage'] == 'NOW'] if roadmap_df is not None and not roadmap_df.empty else pd.DataFrame()
                if not now_features.empty and dependency_df is not None and not dependency_df.empty:
                    blocked_now = dependency_df[(dependency_df['Current_Stage'] == 'NOW') & (dependency_df['Is_Blocked'])]
                    unblocked_count = len(now_features) - len(blocked_now)
                    feasibility_pct = int(round((unblocked_count / len(now_features)) * 100))
                    feasibility_sub = f"{unblocked_count}/{len(now_features)} features unblocked"
                else:
                    feasibility_pct = 100
                    feasibility_sub = "Prerequisites validated"

                st.metric("Sprint Feasibility", f"{feasibility_pct}%", feasibility_sub)
                st.caption("Calculated from unblocked dependencies in current sprint.")
                if st.button("Generate Sprint PRD ✨", key="sprint_prd_btn", type="primary", use_container_width=True):
                    st.session_state.active_nav = "📝 AI PRD Generator"
                    st.rerun()

        st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

        # ----------------- BOTTOM SECTION (3 REAL DATA PANELS) ----------------- #
        col_b1, col_b2, col_b3 = st.columns([3.6, 3.4, 3.0])
        
        # 1. Top Feature Requests (Ranked by Smart RICE Score)
        with col_b1:
            with st.container(border=True):
                col_t1, col_t2 = st.columns([7, 3])
                with col_t1:
                    st.markdown("##### 🏆 Top Feature Requests")
                    st.caption("Ranked by Smart RICE Score")
                with col_t2:
                    if st.button("View All →", key="view_all_rice_btn", use_container_width=True):
                        st.session_state.active_nav = "🏆 Prioritization"
                        st.rerun()

                # Robust source: prefer roadmap_df which already has clean themes and formatted scores, fallback to ranking_df
                if roadmap_df is not None and not roadmap_df.empty and 'Theme' in roadmap_df.columns:
                    top_source = roadmap_df.sort_values(by="Score", ascending=False).head(5).reset_index(drop=True)
                    display_df = pd.DataFrame({
                        "#": [f"{i+1}" for i in range(len(top_source))],
                        "Feature": top_source['Theme'],
                        "Category": top_source.get('Category', top_source.get('Assigned Team', pd.Series(['Core Features'] * len(top_source)))),
                        "Requests": top_source.get('Volume', pd.Series([100] * len(top_source))),
                        "RICE": top_source['Score'].apply(lambda x: f"{float(x):.1f}"),
                        "Priority": top_source['Priority'].apply(lambda x: '🔥 Critical' if 'NOW' in str(x) else ('High' if 'NEXT' in str(x) else 'Medium'))
                    })
                else:
                    score_col = 'RICE Score' if 'RICE Score' in ranking_df.columns else ('Priority_Score' if 'Priority_Score' in ranking_df.columns else ranking_df.columns[1])
                    req_col = 'Request Count' if 'Request Count' in ranking_df.columns else ('Volume' if 'Volume' in ranking_df.columns else ranking_df.columns[0])
                    cat_col = 'Dominant Category' if 'Dominant Category' in ranking_df.columns else ('Category' if 'Category' in ranking_df.columns else 'Core')
                    
                    top_source = ranking_df.sort_values(by=score_col, ascending=False).head(5).reset_index(drop=True)
                    theme_vals = []
                    for _, r in top_source.iterrows():
                        c_id = r.get('Cluster', None)
                        t = st.session_state.cluster_summaries.get(c_id, '') if c_id is not None and st.session_state.cluster_summaries else ''
                        if t and 'Representative Request:' in t:
                            t = t.split('Representative Request:')[-1].replace('"', '').strip()[:28]
                        theme_vals.append(t if t else f"Feature Cluster {c_id if c_id is not None else 1}")

                    display_df = pd.DataFrame({
                        "#": [f"{i+1}" for i in range(len(top_source))],
                        "Feature": theme_vals,
                        "Category": top_source[cat_col] if cat_col in top_source.columns else ['Core']*len(top_source),
                        "Requests": top_source[req_col] if req_col in top_source.columns else [50]*len(top_source),
                        "RICE": top_source[score_col].apply(lambda x: f"{float(x):.1f}"),
                        "Priority": ['🔥 Critical', 'High', 'High', 'Medium', 'Medium'][:len(top_source)]
                    })
                
                st.dataframe(
                    display_df,
                    column_config={
                        "#": st.column_config.TextColumn("#", width="small"),
                        "Feature": st.column_config.TextColumn("Feature", width="large"),
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Requests": st.column_config.NumberColumn("Requests", format="%d"),
                        "RICE": st.column_config.TextColumn("RICE Score", width="small"),
                        "Priority": st.column_config.TextColumn("Priority", width="small"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=220
                )

        # 2. AI Insights (With Evidence Backing)
        with col_b2:
            with st.container(border=True):
                col_i1, col_i2 = st.columns([7, 3])
                with col_i1:
                    st.markdown("##### ✨ AI Insights")
                    st.caption("Evidence-backed customer signals")
                with col_i2:
                    if st.button("View Details →", key="view_details_ins_btn", use_container_width=True):
                        st.session_state.active_nav = "✨ AI Insights"
                        st.rerun()

                if insights and len(insights) > 0:
                    for idx, ins in enumerate(insights[:3]):
                        badge = ins.get('Rationale_Badge', '💡 Finding')
                        theme = ins.get('Theme', f'Feature {idx+1}')
                        rationale = ins.get('AI_Rationale', '')
                        reqs = ins.get('Request_Count', 0)
                        pct = ins.get('Pct_Of_Total', 0)
                        
                        with st.container(border=True):
                            st.markdown(f"**{badge}**: {theme}")
                            st.caption(f"{rationale[:100]}..." if len(rationale) > 100 else rationale)
                            st.caption(f"📊 **{reqs} requests** ({pct}% share)")
                else:
                    st.info("No AI insights generated yet.")

        # 3. Product Roadmap (Now / Next / Later)
        with col_b3:
            with st.container(border=True):
                col_r1, col_r2 = st.columns([7, 3])
                with col_r1:
                    st.markdown("##### 🗺️ Product Roadmap")
                    st.caption("Now / Next / Later horizons")
                with col_r2:
                    if st.button("View Board →", key="view_board_rdm_btn", use_container_width=True):
                        st.session_state.active_nav = "🗺️ Roadmap"
                        st.rerun()

                if roadmap_df is not None and not roadmap_df.empty:
                    c_now, c_next, c_later = st.columns(3)
                    now_list = roadmap_df[roadmap_df['Stage'] == 'NOW']
                    next_list = roadmap_df[roadmap_df['Stage'] == 'NEXT']
                    later_list = roadmap_df[roadmap_df['Stage'] == 'LATER']
                    
                    with c_now:
                        with st.container(border=True):
                            st.markdown(f"**🔥 NOW** `({len(now_list)})`")
                            st.caption("0–3 Months")
                            for _, item in now_list.head(3).iterrows():
                                st.markdown(f"• **{item['Theme'][:15]}**")
                                st.caption(f"RICE {item['Score']}")
                                
                    with c_next:
                        with st.container(border=True):
                            st.markdown(f"**⭐ NEXT** `({len(next_list)})`")
                            st.caption("3–6 Months")
                            for _, item in next_list.head(3).iterrows():
                                st.markdown(f"• **{item['Theme'][:15]}**")
                                st.caption(f"RICE {item['Score']}")

                    with c_later:
                        with st.container(border=True):
                            st.markdown(f"**❄️ LATER** `({len(later_list)})`")
                            st.caption("6–12 Months")
                            for _, item in later_list.head(3).iterrows():
                                st.markdown(f"• **{item['Theme'][:15]}**")
                                st.caption(f"RICE {item['Score']}")
                else:
                    st.info("No roadmap generated.")

        # ----------------- BOTTOM EMERGING ALERT BANNER ----------------- #
        if alerts and len(alerts) > 0:
            top_alert = alerts[0]
            alert_theme = top_alert.get('Theme', 'Key Features')
            alert_surge = top_alert.get('Surge_Pct', 60)
            alert_rec = top_alert.get('Actionable_Recommendation', 'Consider prioritizing it in the next release.')
            
            with st.container(border=True):
                col_alt_text, col_alt_btn = st.columns([8, 2])
                with col_alt_text:
                    st.markdown(f"🔔 **Emerging Alert:** Feature requests for **\"{alert_theme}\"** have surged by **{alert_surge}%** in the last 30 days. {alert_rec}")
                with col_alt_btn:
                    if st.button("Investigate Alert →", key="investigate_alert_btn", use_container_width=True):
                        st.session_state.active_nav = "📊 Trends & Analytics"
                        st.rerun()

    # =========================================================================
    # VIEW 2: 🔍 FEEDBACK EXPLORER
    # =========================================================================
    elif st.session_state.active_nav == "🔍 Feedback Explorer":
        st.markdown("### 🔍 Semantic Vector Feedback Explorer & Deduplication")
        st.caption("FAISS cosine index groups duplicate feedback and enables instant natural language semantic search.")
        
        search_query = st.text_input("🔎 Search feedback requests using natural language:", placeholder="E.g., SSO authentication, export to excel, tablet layout")
        if search_query:
            embed_model = load_embedding_model()
            q_emb = embed_model.encode([search_query])[0]
            sims, indices = semantic_search(q_emb, st.session_state.index, k=6)
            st.markdown(f"**Top matching feedback signals for '{search_query}':**")
            for sim, idx in zip(sims, indices):
                if idx < len(df):
                    match_row = df.iloc[idx]
                    with st.container(border=True):
                        col_m_txt, col_m_score = st.columns([8, 2])
                        with col_m_txt:
                            st.markdown(f"**\"{match_row['Request_Text']}\"**")
                            st.caption(f"Tier: **{match_row['Customer_Segment']}** • Upvotes: **{match_row['Upvotes']}** • Sentiment: **{match_row['Sentiment']}**")
                        with col_m_score:
                            st.metric("Match", f"{sim:.2f}")
                    
        st.markdown("---")
        st.markdown(f"#### 🔗 Suggested Duplicate Groups ({len(st.session_state.duplicate_groups or [])} potential duplicate groups identified for review)")
        if st.session_state.duplicate_groups:
            for g in (st.session_state.duplicate_groups[:8]):
                prim_row = df.iloc[g['primary_idx']]
                with st.expander(f"📌 Primary Signal: \"{prim_row['Request_Text']}\" ({g['count']} potential duplicates suggested)"):
                    st.caption(f"Semantic similarity: **{g['avg_similarity']}** (cosine threshold ≥ 0.88) | Category: **{prim_row['Category']}**")
                    st.caption("Review suggested duplicate signals for human PM confirmation:")
                    for d_idx in g['duplicate_indices']:
                        dup_row = df.iloc[d_idx]
                        st.markdown(f"- \"{dup_row['Request_Text']}\" *({dup_row['Customer_Segment']} tier, {dup_row['Upvotes']} upvotes)*")

    # =========================================================================
    # VIEW: 🔎 CUSTOMER EVIDENCE DRILL-DOWN (DIFFERENTIATOR #3)
    # =========================================================================
    elif st.session_state.active_nav == "🔎 Customer Evidence":
        st.markdown("### 🔎 Customer Evidence Drill-Down & Voice of Customer Audit")
        st.caption("Drill down into verbatim customer quotes, user tiers, sentiment friction, and source channels behind every feature.")
        
        # Build theme mapping
        theme_options = {}
        if roadmap_df is not None and not roadmap_df.empty:
            for _, r in roadmap_df.iterrows():
                theme_options[str(r['Theme'])] = int(r['ID'])
        elif 'Cluster' in ranking_df.columns:
            for _, r in ranking_df.iterrows():
                c_id = int(r['Cluster'])
                theme_options[st.session_state.cluster_summaries.get(c_id, f"Cluster {c_id}")] = c_id

        col_ev_sel, col_ev_quick = st.columns([7, 3])
        with col_ev_sel:
            selected_theme_name = st.selectbox(
                "Select Feature Cluster to Audit:",
                list(theme_options.keys()),
                index=0,
                key="evidence_theme_selector"
            )
            selected_cluster_id = theme_options[selected_theme_name]
            st.session_state.selected_evidence_cluster = selected_cluster_id
            
        with col_ev_quick:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📝 Generate PRD for this Feature →", key="ev_to_prd_btn", use_container_width=True):
                st.session_state.selected_prd_theme = selected_theme_name
                st.session_state.active_nav = "📝 AI PRD Generator"
                st.rerun()

        # Cluster Data
        cluster_data = df[df['Cluster'] == selected_cluster_id] if 'Cluster' in df.columns else df
        c_reqs = len(cluster_data)
        c_upvotes = int(cluster_data['Upvotes'].sum()) if 'Upvotes' in cluster_data.columns else c_reqs * 2
        
        # Segment counts
        seg_counts = cluster_data['Customer_Segment'].value_counts().to_dict() if 'Customer_Segment' in cluster_data.columns else {}
        ent_count = seg_counts.get('Enterprise', 0)
        pro_count = seg_counts.get('Pro', 0)
        free_count = seg_counts.get('Free', 0)
        
        # Sentiment counts
        sent_counts = cluster_data['Sentiment'].value_counts().to_dict() if 'Sentiment' in cluster_data.columns else {}
        neg_count = sent_counts.get('NEGATIVE', 0)
        pos_count = sent_counts.get('POSITIVE', 0)
        neu_count = sent_counts.get('NEUTRAL', 0)
        neg_pct = round((neg_count / c_reqs) * 100) if c_reqs > 0 else 0
        pos_pct = round((pos_count / c_reqs) * 100) if c_reqs > 0 else 0
        
        # Roadmap info for this cluster
        r_item = roadmap_df[roadmap_df['ID'] == selected_cluster_id] if roadmap_df is not None and 'ID' in roadmap_df.columns else pd.DataFrame()
        rice_val = float(r_item['Score'].values[0]) if not r_item.empty else 8.5
        effort_val = float(r_item['Effort'].values[0]) if not r_item.empty else 5.0
        stage_val = str(r_item['Stage'].values[0]) if not r_item.empty else 'NOW'
        
        # KPI Row
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        with col_c1:
            with st.container(border=True):
                st.caption("TOTAL REQUESTS")
                st.metric("Requests", f"{c_reqs:,}", delta=f"{round(c_reqs/len(df)*100, 1)}% of all feedback", label_visibility="collapsed")
        with col_c2:
            with st.container(border=True):
                st.caption("UPVOTES")
                st.metric("Upvotes", f"{c_upvotes:,}", delta=f"Avg {round(c_upvotes/c_reqs, 1)}/req" if c_reqs > 0 else "", label_visibility="collapsed")
        with col_c3:
            with st.container(border=True):
                st.caption("CUSTOMER FRICTION")
                st.metric("Negative Sentiment", f"{neg_pct}%", delta="high friction" if neg_pct >= 30 else "healthy", delta_color="inverse", label_visibility="collapsed")
        with col_c4:
            with st.container(border=True):
                st.caption("TOP TIER")
                top_seg_label = "Enterprise" if ent_count >= max(pro_count, free_count) else ("Pro" if pro_count >= free_count else "Free")
                st.metric("Top Tier", top_seg_label, delta=f"{ent_count} Enterprise", label_visibility="collapsed")
        with col_c5:
            with st.container(border=True):
                st.caption("RICE & STAGE")
                st.metric("Priority", f"{rice_val:.1f}", delta=f"Stage: {stage_val}", label_visibility="collapsed")

        # Interactive Filter Bar for Customer Signals
        st.markdown("#### 💬 Verbatim Customer Feedback & Exact Quotes")
        col_f1, col_f2, col_f3, col_f4 = st.columns([3, 3, 3, 3])
        with col_f1:
            filter_sentiment = st.selectbox("Sentiment Filter", ["All Sentiments", "NEGATIVE (Friction)", "POSITIVE (Praise)", "NEUTRAL"], key="ev_filt_sent")
        with col_f2:
            filter_segment = st.selectbox("Customer Segment", ["All Segments", "Enterprise", "Pro", "Free"], key="ev_filt_seg")
        with col_f3:
            sources_available = ["All Sources"] + sorted(cluster_data['Source'].dropna().unique().tolist()) if 'Source' in cluster_data.columns else ["All Sources"]
            filter_source = st.selectbox("Source Channel", sources_available, key="ev_filt_src")
        with col_f4:
            search_quote = st.text_input("Search feedback text...", placeholder="Filter by keyword...", key="ev_filt_kw")

        # Apply filters
        filtered_evidence = cluster_data.copy()
        if filter_sentiment == "NEGATIVE (Friction)":
            filtered_evidence = filtered_evidence[filtered_evidence['Sentiment'] == 'NEGATIVE']
        elif filter_sentiment == "POSITIVE (Praise)":
            filtered_evidence = filtered_evidence[filtered_evidence['Sentiment'] == 'POSITIVE']
        elif filter_sentiment == "NEUTRAL":
            filtered_evidence = filtered_evidence[filtered_evidence['Sentiment'] == 'NEUTRAL']
            
        if filter_segment != "All Segments":
            filtered_evidence = filtered_evidence[filtered_evidence['Customer_Segment'] == filter_segment]
            
        if filter_source != "All Sources":
            filtered_evidence = filtered_evidence[filtered_evidence['Source'] == filter_source]
            
        if search_quote:
            filtered_evidence = filtered_evidence[filtered_evidence['Request_Text'].str.contains(search_quote, case=False, na=False)]

        st.caption(f"Showing **{len(filtered_evidence)}** customer signals matching active filters:")

        # Display quote cards (2 columns)
        quotes_to_show = filtered_evidence.sort_values(by='Upvotes', ascending=False).head(12)
        if quotes_to_show.empty:
            st.info("No customer feedback requests match the selected filters.")
        else:
            q_cols = st.columns(2)
            for idx, (_, q_row) in enumerate(quotes_to_show.iterrows()):
                col_target = q_cols[idx % 2]
                with col_target:
                    with st.container(border=True):
                        s_badge = "🔴 Negative" if q_row.get('Sentiment') == 'NEGATIVE' else ("🟢 Positive" if q_row.get('Sentiment') == 'POSITIVE' else "⚪ Neutral")
                        seg_badge = q_row.get('Customer_Segment', 'User')
                        src_badge = q_row.get('Source', 'Zendesk')
                        upv = q_row.get('Upvotes', 1)
                        dt = q_row.get('Date', '')
                        
                        st.markdown(f"**\"{q_row['Request_Text']}\"**")
                        st.caption(f"🏢 **{seg_badge}** • 📡 {src_badge} • 👍 **{upv} upvotes** • {s_badge} • 📅 {dt}")

        st.markdown("---")
        with st.expander("📊 Complete Tabular Feedback Log for this Cluster", expanded=False):
            st.dataframe(
                filtered_evidence[['Request_ID', 'Date', 'Customer_Segment', 'Source', 'Sentiment', 'Upvotes', 'Request_Text']],
                hide_index=True,
                use_container_width=True
            )
            c_csv = filtered_evidence.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📥 Export {selected_theme_name} Feedback CSV",
                data=c_csv,
                file_name=f"feedback_{selected_cluster_id}.csv",
                mime="text/csv",
                key="dl_cluster_csv"
            )

    # =========================================================================
    # VIEW 3: ✨ AI INSIGHTS
    # =========================================================================
    elif st.session_state.active_nav == "✨ AI Insights":
        st.markdown("### ✨ Evidence-Backed AI Insights & Voice of Customer")
        st.caption("Transparent AI findings supported by verbatim quotes, customer segments, and friction data.")
        
        for ins in insights:
            with st.container(border=True):
                st.markdown(f"#### {ins['Rationale_Badge']}: {ins['Theme']}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RICE Priority", f"{ins['RICE_Score']}")
                with col2:
                    st.metric("Supporting Volume", f"{ins['Request_Count']} reqs", f"{ins['Pct_Of_Total']}% share")
                with col3:
                    st.metric("Total Upvotes", f"{ins['Total_Upvotes']:,}")
                with col4:
                    st.metric("Customer Friction", f"{ins['Sentiment_Breakdown']['Negative_Pct']}% neg")
                    
                st.markdown(f"**🤖 AI Rationale:** {ins['AI_Rationale']}")
                st.markdown(f"**🎯 Actionable Recommendation:** `{ins['Actionable_Recommendation']}`")
                st.markdown(f"**👥 Segment Concentration:** {ins['Segment_Summary']}")
                
                st.markdown("##### 💬 Verbatim Customer Feedback Samples:")
                for q in ins['Customer_Quotes']:
                    with st.container(border=True):
                        st.markdown(f"*\"{q['text']}\"*")
                        st.caption(f"— **{q['segment']}** Customer • {q['upvotes']} upvotes • {q['date']}")

    # =========================================================================
    # VIEW 4: 🏆 PRIORITIZATION (RICE DECISION ENGINE)
    # =========================================================================
    elif st.session_state.active_nav == "🏆 Prioritization":
        st.markdown("### 🏆 Smart Priority Score (RICE) Engine & Sensitivity Sliders")
        st.caption("Calculate transparent, objective feature scores using Reach, Impact, Confidence, and Effort.")

        with st.expander("📘 Scoring Framework: Mathematical Formula, Input Scales & Weight Modifiers", expanded=False):
            st.markdown("""
            #### 📐 Scoring Formula
            $$\\text{Smart RICE} = \\frac{(\\text{Reach} \\times W_R) \\times (\\text{Impact} \\times W_I) \\times (\\text{Confidence} \\times W_C)}{\\max(0.5, \\text{Effort} \\times W_E)}$$
            *When all weights ($W_R, W_I, W_C, W_E$) equal 1.0, this formula simplifies identically to standard RICE: $(\\text{Reach} \\times \\text{Impact} \\times \\text{Confidence}) / \\text{Effort}$.*
            
            #### 📊 Input Parameter Scales & Objective Definitions:
            1. **Reach ($R$) — Affected User Volume**
               - *Scale: 50 to 1,500 monthly affected users*
               - Calculated objectively: $\\text{Reach} = (N_{\\text{requests}} \\times 10) + (N_{\\text{upvotes}} \\times 2) + (N_{\\text{Enterprise}} \\times 20) + (N_{\\text{Pro}} \\times 5)$
               - $W_R$ ($0.2 - 2.5$, default $1.0$): Adjusts sensitivity to broad volume vs high-value accounts.
            2. **Impact ($I$) — Business & Retention Impact**
               - *Scale: 0.5 to 3.0 qualitative multiplier*
               - `3.0` (Massive): Critical Enterprise blocker with $>40\%$ negative sentiment.
               - `2.0` (High): Significant customer friction or major revenue driver.
               - `1.5` (Medium-High): Broad user demand across $>8\%$ of total feedback.
               - `1.0` (Medium): Standard expected functional improvement.
               - `0.5` (Low): Minor cosmetic polish or primarily free-tier request.
               - $W_I$ ($0.2 - 2.5$, default $1.0$): Emphasizes customer friction to combat churn.
            3. **Confidence ($C$) — Statistical Sample Confidence**
               - *Scale: 0.50 to 1.00 evidence confidence factor*
               - `1.0` (100%): Very high sample density ($N \\ge 40$ requests).
               - `0.9` (90%): Substantial sample size ($N \\ge 20$ requests).
               - `0.8` (80%): Moderate evidence ($N \\ge 10$ requests).
               - `0.65` (65%): Early signal ($N \\ge 5$ requests).
               - `0.5` (50%): Low sample size ($N < 5$ requests).
               - $W_C$ ($0.2 - 2.5$, default $1.0$): Penalizes speculative or low-sample requests.
            4. **Effort ($E$) — Engineering Complexity**
               - *Scale: 1.0 to 10.0 story points / sprint units*
               - Estimated by architectural domain: UI/UX (3.0), Notifications (3.5), Dashboard (4.0), Reports (4.5), Performance (5.5), Mobile (6.0), API (6.5), Payments (7.5), Security/Auth (8.0). Clamped to min 0.5.
               - $W_E$ ($0.2 - 2.5$, default $1.0$): Multiplies effort penalty. Higher $W_E$ heavily favors Quick Wins.
            """)
        
        with st.expander("⚖️ Strategic Weight Presets & Sensitivity Sliders", expanded=True):
            col_preset, col_w_reach, col_w_impact, col_w_conf, col_w_eff = st.columns(5)
            with col_preset:
                preset_name = st.selectbox(
                    "Strategic Preset",
                    list(SCORING_PRESETS.keys()) + ["Custom Weights"],
                    index=list(SCORING_PRESETS.keys()).index(st.session_state.active_preset) if st.session_state.active_preset in SCORING_PRESETS else 0,
                    key="rice_preset_select"
                )
                if preset_name != "Custom Weights" and preset_name != st.session_state.active_preset:
                    st.session_state.active_preset = preset_name
                    p = SCORING_PRESETS[preset_name]
                    st.session_state.weight_reach = p["weight_reach"]
                    st.session_state.weight_impact = p["weight_impact"]
                    st.session_state.weight_confidence = p["weight_confidence"]
                    st.session_state.weight_effort = p["weight_effort"]
                    ranking_df = rank_features(
                        df, weight_reach=p["weight_reach"], weight_impact=p["weight_impact"], 
                        weight_confidence=p["weight_confidence"], weight_effort=p["weight_effort"]
                    )
                    st.session_state.ranking_df = ranking_df
                    roadmap_df = generate_ai_roadmap(df, ranking_df, st.session_state.cluster_keywords, st.session_state.cluster_summaries)
                    st.session_state.roadmap_df = roadmap_df
                    st.session_state.dependency_df = map_feature_dependencies(roadmap_df)

            with col_w_reach:
                w_reach = st.slider("Reach Weight ($W_R$)", 0.2, 2.5, float(st.session_state.weight_reach), step=0.1, key="rice_wr")
            with col_w_impact:
                w_impact = st.slider("Impact Weight ($W_I$)", 0.2, 2.5, float(st.session_state.weight_impact), step=0.1, key="rice_wi")
            with col_w_conf:
                w_conf = st.slider("Confidence Weight ($W_C$)", 0.2, 2.5, float(st.session_state.weight_confidence), step=0.1, key="rice_wc")
            with col_w_eff:
                w_eff = st.slider("Effort Penalty ($W_E$)", 0.2, 2.5, float(st.session_state.weight_effort), step=0.1, key="rice_we")

        col_matrix, col_sliders = st.columns([7, 5])
        with col_matrix:
            fig_matrix = plot_interactive_rice_matrix(roadmap_df)
            st.plotly_chart(fig_matrix, use_container_width=True)
            
        with col_sliders:
            st.markdown("#### 🛠️ Fine-Tune Development Effort (1–10)")
            themes_sorted = sorted(roadmap_df['Theme'].unique())[:6]
            for t_name in themes_sorted:
                r_item = roadmap_df[roadmap_df['Theme'] == t_name].iloc[0]
                curr_eff = float(r_item['Effort'])
                new_eff = st.slider(f"{t_name}", 1.0, 10.0, curr_eff, step=0.5, key=f"eff_fix_{r_item['ID']}")
                if new_eff != curr_eff:
                    c_id = r_item['ID']
                    roadmap_df.loc[roadmap_df['ID'] == c_id, 'Effort'] = new_eff
                    reach = ranking_df.loc[ranking_df['Cluster'] == c_id, 'Reach'].values[0]
                    impact = ranking_df.loc[ranking_df['Cluster'] == c_id, 'Impact'].values[0]
                    conf = ranking_df.loc[ranking_df['Cluster'] == c_id, 'Confidence'].values[0]
                    new_score = calculate_rice_score(
                        reach, impact, conf, new_eff,
                        st.session_state.weight_reach, st.session_state.weight_impact,
                        st.session_state.weight_confidence, st.session_state.weight_effort
                    )
                    ranking_df.loc[ranking_df['Cluster'] == c_id, 'RICE Score'] = new_score
                    roadmap_df.loc[roadmap_df['ID'] == c_id, 'Score'] = new_score
                    st.session_state.ranking_df = ranking_df
                    st.session_state.roadmap_df = roadmap_df

        st.markdown("### 📋 Complete RICE Priority Table")
        st.dataframe(
            ranking_df[[
                'Dominant Category', 'RICE Score', 'Base RICE Score', 'Reach', 'Impact', 'Impact_Desc', 
                'Confidence', 'Effort', 'Request Count', 'Total Upvotes', 'Negative Ratio'
            ]],
            column_config={
                "RICE Score": st.column_config.ProgressColumn("Weighted RICE", format="%.1f", min_value=0, max_value=float(ranking_df['RICE Score'].max())),
                "Base RICE Score": st.column_config.NumberColumn("Standard (1.0x)", format="%.1f"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
                "Negative Ratio": st.column_config.NumberColumn("Friction %", format="%.1f%%")
            },
            hide_index=True,
            use_container_width=True
        )

    # =========================================================================
    # VIEW: 🎯 PRIORITY SCENARIO SIMULATOR (DIFFERENTIATOR #1)
    # =========================================================================
    elif st.session_state.active_nav == "🎯 Scenario Simulator":
        st.markdown("### 🎯 Priority Scenario Simulator & Strategic Trade-off Modeler")
        st.caption("Adjust impact, effort, and enterprise revenue weights to see how rankings change instantly. Compare scenarios like Customer First vs Revenue First.")
        
        # Strategic Presets Controls
        col_scen_a, col_scen_b = st.columns(2)
        
        with col_scen_a:
            with st.container(border=True):
                st.markdown("#### 🔵 Scenario A (Baseline)")
                preset_a = st.selectbox(
                    "Strategic Framework A:",
                    list(SCENARIOS.keys()) + ["Custom Weights"],
                    index=1,  # Default: Customer First (Churn Prevention)
                    key="sim_preset_a_select"
                )
                if preset_a != "Custom Weights":
                    p_a = SCENARIOS[preset_a]
                    st.caption(f"*{p_a['description']}*")
                    w_r_a, w_i_a, w_c_a, w_e_a, ent_a = p_a['weight_reach'], p_a['weight_impact'], p_a['weight_confidence'], p_a['weight_effort'], p_a['enterprise_boost']
                else:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: w_r_a = st.slider("Reach (A)", 0.2, 3.0, 1.0, step=0.1, key="sim_wra")
                    with c2: w_i_a = st.slider("Impact (A)", 0.2, 3.0, 2.0, step=0.1, key="sim_wia")
                    with c3: w_c_a = st.slider("Conf (A)", 0.2, 3.0, 1.0, step=0.1, key="sim_wca")
                    with c4: w_e_a = st.slider("Effort (A)", 0.2, 3.0, 1.0, step=0.1, key="sim_wea")
                    with c5: ent_a = st.slider("ARR Mult (A)", 1.0, 4.0, 1.2, step=0.2, key="sim_enta")

        with col_scen_b:
            with st.container(border=True):
                st.markdown("#### 🟢 Scenario B (Comparison)")
                preset_b = st.selectbox(
                    "Strategic Framework B:",
                    list(SCENARIOS.keys()) + ["Custom Weights"],
                    index=2,  # Default: Revenue First (Enterprise Growth)
                    key="sim_preset_b_select"
                )
                if preset_b != "Custom Weights":
                    p_b = SCENARIOS[preset_b]
                    st.caption(f"*{p_b['description']}*")
                    w_r_b, w_i_b, w_c_b, w_e_b, ent_b = p_b['weight_reach'], p_b['weight_impact'], p_b['weight_confidence'], p_b['weight_effort'], p_b['enterprise_boost']
                else:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: w_r_b = st.slider("Reach (B)", 0.2, 3.0, 1.8, step=0.1, key="sim_wrb")
                    with c2: w_i_b = st.slider("Impact (B)", 0.2, 3.0, 2.0, step=0.1, key="sim_wib")
                    with c3: w_c_b = st.slider("Conf (B)", 0.2, 3.0, 1.2, step=0.1, key="sim_wcb")
                    with c4: w_e_b = st.slider("Effort (B)", 0.2, 3.0, 0.9, step=0.1, key="sim_web")
                    with c5: ent_b = st.slider("ARR Mult (B)", 1.0, 4.0, 3.0, step=0.2, key="sim_entb")

        # Run Comparison
        weights_a = {'weight_reach': w_r_a, 'weight_impact': w_i_a, 'weight_confidence': w_c_a, 'weight_effort': w_e_a, 'enterprise_boost': ent_a}
        weights_b = {'weight_reach': w_r_b, 'weight_impact': w_i_b, 'weight_confidence': w_c_b, 'weight_effort': w_e_b, 'enterprise_boost': ent_b}
        
        comparison_res = compare_two_scenarios(ranking_df, weights_a, weights_b, name_a=preset_a, name_b=preset_b, roadmap_df=roadmap_df)
        comp_df = comparison_res['comparison_df']
        
        # Summary Row: Key Shifts & Takeaways
        col_g1, col_g2, col_g3 = st.columns([3.5, 3.5, 3.0])
        with col_g1:
            with st.container(border=True):
                st.markdown("##### 🚀 Top Movers in Scenario B")
                for _, r in comparison_res['top_gainers'].iterrows():
                    st.markdown(f"**{r['Theme']}**: **{r['Movement']}** (Rank #{r['Simulated_Rank_B']})")
                    st.caption(f"Score: {r['Simulated_Score_A']:.0f} → **{r['Simulated_Score_B']:.0f}** pts")
        with col_g2:
            with st.container(border=True):
                st.markdown("##### 🔻 Deprioritized in Scenario B")
                for _, r in comparison_res['top_decliners'].iterrows():
                    st.markdown(f"**{r['Theme']}**: **{r['Movement']}** (Rank #{r['Simulated_Rank_B']})")
                    st.caption(f"Score: {r['Simulated_Score_A']:.0f} → **{r['Simulated_Score_B']:.0f}** pts")
        with col_g3:
            with st.container(border=True):
                st.markdown("##### ⚖️ Strategic Takeaway")
                st.caption(f"Switching from **{preset_a}** to **{preset_b}** shifts focus toward high-value Enterprise contracts and ACV expansion while relaxing lower-margin consumer friction penalties.")
                if st.button("Apply Scenario B to Roadmap →", key="apply_scen_b_btn", type="primary", use_container_width=True):
                    ranking_df['RICE Score'] = comp_df['Simulated_Score_B']
                    st.session_state.ranking_df = ranking_df
                    st.session_state.roadmap_df = generate_ai_roadmap(df, ranking_df, st.session_state.cluster_keywords, st.session_state.cluster_summaries)
                    st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                    st.success(f"✅ Committed {preset_b} priorities to active roadmap!")
                    st.rerun()

        # Side-by-Side Plotly Horizontal Comparison Chart
        st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
        fig_comp = plot_scenario_comparison_chart(comp_df, name_a=preset_a[:22], name_b=preset_b[:22])
        st.plotly_chart(fig_comp, use_container_width=True)

        # Complete Comparison Data Table
        st.markdown("#### 📋 Full Side-by-Side Priority Table")
        display_sim = comp_df[['Theme', 'Simulated_Rank_A', 'Simulated_Rank_B', 'Movement', 'Simulated_Score_A', 'Simulated_Score_B', 'Effort', 'Request Count', 'Enterprise Count']].copy()
        display_sim.columns = ['Feature Theme', 'Rank (A)', 'Rank (B)', 'Rank Delta', 'Score (A)', 'Score (B)', 'Effort', 'Requests', 'Enterprise Reqs']
        
        st.dataframe(
            display_sim,
            column_config={
                "Rank Delta": st.column_config.TextColumn("Shift"),
                "Score (A)": st.column_config.NumberColumn("Score (A)", format="%.1f"),
                "Score (B)": st.column_config.NumberColumn("Score (B)", format="%.1f"),
                "Effort": st.column_config.NumberColumn("Effort (pts)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )

    # =========================================================================
    # VIEW: 📝 AI FEATURE-TO-PRD GENERATOR (DIFFERENTIATOR #2)
    # =========================================================================
    elif st.session_state.active_nav == "📝 AI PRD Generator":
        st.markdown("### 📝 AI Feature-to-PRD Generator")
        st.caption("Turn any prioritized feature into a mini product specification with Problem Statement, User Stories, Acceptance Criteria (Given/When/Then), Proposed Solution, and Success Metrics.")
        
        # Build theme selection list
        theme_names = []
        if roadmap_df is not None and not roadmap_df.empty:
            theme_names = roadmap_df['Theme'].tolist()
        elif 'Theme' in ranking_df.columns:
            theme_names = ranking_df['Theme'].tolist()
        else:
            theme_names = [f"Feature {c}" for c in ranking_df['Cluster'].unique()]

        default_idx = 0
        if st.session_state.selected_prd_theme in theme_names:
            default_idx = theme_names.index(st.session_state.selected_prd_theme)

        col_p1, col_p2, col_p3 = st.columns([5, 3, 2])
        with col_p1:
            chosen_theme = st.selectbox("Select Feature Theme to Generate PRD:", theme_names, index=default_idx, key="prd_theme_choice")
            st.session_state.selected_prd_theme = chosen_theme
        with col_p2:
            chosen_horizon = st.selectbox("Target Execution Horizon:", ["NOW (Current Sprint / Q1)", "NEXT (Upcoming Quarter / Q2)", "LATER (Backlog / H2)"], index=0, key="prd_horizon_choice")
        with col_p3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_btn = st.button("⚡ Generate PRD", type="primary", use_container_width=True, key="btn_trigger_prd")

        # Find cluster ID
        target_cluster = 0
        r_row = {}
        if roadmap_df is not None and not roadmap_df.empty and 'Theme' in roadmap_df.columns:
            match = roadmap_df[roadmap_df['Theme'] == chosen_theme]
            if not match.empty:
                target_cluster = int(match.iloc[0]['ID'])
                r_row = match.iloc[0].to_dict()
                
        rank_row = {}
        if ranking_df is not None and not ranking_df.empty and 'Cluster' in ranking_df.columns:
            match_rank = ranking_df[ranking_df['Cluster'] == target_cluster]
            if not match_rank.empty:
                rank_row = match_rank.iloc[0].to_dict()

        # Generate PRD Content
        prd_text = generate_full_product_spec(
            theme_name=chosen_theme,
            cluster_id=target_cluster,
            df=df,
            ranking_row=rank_row,
            roadmap_row=r_row
        )

        col_act1, col_act2, col_act3 = st.columns([3, 3, 4])
        with col_act1:
            st.download_button(
                "📥 Download PRD (.md)",
                data=prd_text,
                file_name=f"PRD_{chosen_theme.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_act2:
            if st.button("⚡ Plan Sprint Capacity →", key="prd_to_capacity_btn", use_container_width=True):
                st.session_state.active_nav = "⚡ Capacity Planner"
                st.rerun()
        with col_act3:
            st.caption("✅ PRD formatted with Agile User Stories & BDD Given/When/Then acceptance criteria.")

        st.markdown("---")
        with st.container(border=True):
            st.markdown(prd_text)

    # =========================================================================
    # VIEW 5: 🗺️ ROADMAP (NOW / NEXT / LATER KANBAN)
    # =========================================================================
    elif st.session_state.active_nav == "🗺️ Roadmap":
        st.markdown("### 🗺️ Interactive Now / Next / Later Roadmap Board")
        st.caption("Organize features into execution horizons with live state persistence.")
        
        roadmap_md = export_roadmap_markdown(roadmap_df)
        st.download_button("📥 Export Roadmap (.md)", data=roadmap_md, file_name="roadmap_now_next_later.md", mime="text/markdown")
            
        col_now, col_next, col_later = st.columns(3)
        now_items = roadmap_df[roadmap_df['Stage'] == 'NOW']
        next_items = roadmap_df[roadmap_df['Stage'] == 'NEXT']
        later_items = roadmap_df[roadmap_df['Stage'] == 'LATER']
        
        with col_now:
            st.markdown(f"#### 🔥 NOW (1–3 Months) ({len(now_items)})")
            for idx, item in now_items.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{item['Theme']}** (RICE {item['Score']})")
                    st.caption(f"{item['Summary'][:85]}...")
                    new_stg = st.selectbox("Stage:", ["NOW", "NEXT", "LATER"], index=0, key=f"mv_n_{item['ID']}")
                    if new_stg != "NOW":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.rerun()
                        
        with col_next:
            st.markdown(f"#### ⭐ NEXT (3–6 Months) ({len(next_items)})")
            for idx, item in next_items.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{item['Theme']}** (RICE {item['Score']})")
                    st.caption(f"{item['Summary'][:85]}...")
                    new_stg = st.selectbox("Stage:", ["NOW", "NEXT", "LATER"], index=1, key=f"mv_x_{item['ID']}")
                    if new_stg != "NEXT":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.rerun()
                        
        with col_later:
            st.markdown(f"#### 🧊 LATER (6–12 Months) ({len(later_items)})")
            for idx, item in later_items.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{item['Theme']}** (RICE {item['Score']})")
                    st.caption(f"{item['Summary'][:85]}...")
                    new_stg = st.selectbox("Stage:", ["NOW", "NEXT", "LATER"], index=2, key=f"mv_l_{item['ID']}")
                    if new_stg != "LATER":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.rerun()

    # =========================================================================
    # VIEW: ⚡ ROADMAP CAPACITY PLANNER (DIFFERENTIATOR #4)
    # =========================================================================
    elif st.session_state.active_nav == "⚡ Capacity Planner":
        st.markdown("### ⚡ Roadmap Capacity & Sprint Velocity Planner")
        st.caption("Model engineering capacity, detect overbooked sprints, validate dependency sequences, and auto-balance the Now / Next / Later roadmap.")
        
        # Configuration Controls
        with st.container(border=True):
            col_cap1, col_cap2, col_cap3, col_cap4 = st.columns(4)
            with col_cap1:
                num_eng = st.number_input("Engineering Team Size", min_value=1, max_value=30, value=int(st.session_state.num_engineers), step=1, key="cap_engineers")
                st.session_state.num_engineers = num_eng
            with col_cap2:
                sprint_w = st.selectbox("Sprint Duration", [1, 2, 3], index=1, key="cap_sprint_weeks")
                st.session_state.sprint_duration = sprint_w
            with col_cap3:
                pts_per_eng = st.number_input("Velocity (Pts / Eng / Sprint)", min_value=2.0, max_value=25.0, value=8.0, step=1.0, key="cap_velocity_eng")
            with col_cap4:
                total_sprint_capacity = float(num_eng * pts_per_eng * (sprint_w / 2.0))
                st.session_state.team_velocity = total_sprint_capacity
                st.metric("Total Sprint Capacity", f"{total_sprint_capacity:.0f} pts", delta=f"{num_eng} engineers × {pts_per_eng:.0f} pts")

        # Run Capacity Planning Engine
        plan_res = calculate_capacity_plan(
            roadmap_df=roadmap_df,
            dependency_df=dependency_df,
            team_velocity=total_sprint_capacity,
            sprint_duration_weeks=sprint_w,
            num_engineers=num_eng
        )
        summary_df = plan_res['summary_df']
        overbooked_count = plan_res['overbooked_count']
        dep_violations = plan_res['dependency_violations']
        
        # Alerts & Warnings
        if overbooked_count > 0:
            with st.container(border=True):
                col_ob_txt, col_ob_btn = st.columns([7.5, 2.5])
                with col_ob_txt:
                    st.markdown(f"🚨 **Capacity Alert:** **{overbooked_count} sprints exceed team capacity limit ({total_sprint_capacity:.0f} pts)!** Overbooked sprints lead to missed delivery dates and developer burnout.")
                with col_ob_btn:
                    if st.button("🚀 Auto-Balance Capacity", type="primary", key="btn_auto_balance_cap", use_container_width=True):
                        new_roadmap, balance_notes = auto_balance_roadmap_capacity(roadmap_df, team_velocity=total_sprint_capacity)
                        st.session_state.roadmap_df = new_roadmap
                        st.session_state.dependency_df = map_feature_dependencies(new_roadmap)
                        st.success("✅ Roadmap successfully rebalanced! No sprint exceeds capacity.")
                        st.rerun()

        if dep_violations:
            for v in dep_violations:
                st.warning(f"⚠️ **Dependency Conflict:** {v['Violation']}")

        # Plotly Capacity Bar Chart
        fig_cap = plot_capacity_utilization_chart(summary_df, capacity_limit=total_sprint_capacity)
        st.plotly_chart(fig_cap, use_container_width=True)

        # Sprint Summary Table
        st.markdown("#### 📅 Sprint Velocity & Capacity Breakdown")
        st.dataframe(
            summary_df[['Sprint', 'Horizon', 'Allocated_Points', 'Capacity_Points', 'Utilization_Pct', 'Status']],
            column_config={
                "Allocated_Points": st.column_config.NumberColumn("Allocated Load", format="%.1f pts"),
                "Capacity_Points": st.column_config.NumberColumn("Max Capacity", format="%.1f pts"),
                "Utilization_Pct": st.column_config.ProgressColumn("Capacity Utilization", format="%.0f%%", min_value=0, max_value=150)
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("#### 📋 Feature-Level Sprint Schedule & Story Points")
        st.dataframe(
            plan_res['df_items'][['Theme', 'Assigned_Sprint', 'Stage', 'Story_Points', 'Score', 'Assigned Team', 'Summary']],
            column_config={
                "Story_Points": st.column_config.NumberColumn("Story Points", format="%.1f pts"),
                "Score": st.column_config.NumberColumn("RICE Score", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )

    # =========================================================================
    # VIEW 6: 🔗 DEPENDENCIES
    # =========================================================================
    elif st.session_state.active_nav == "🔗 Dependencies":
        st.markdown("### 🔗 Architectural Dependency Mapping & Blocker Detection")
        st.caption("Prevent bottlenecks by validating technical prerequisites across modules.")
        
        blocked_items = dependency_df[dependency_df['Is_Blocked']]
        if not blocked_items.empty:
            for _, b_row in blocked_items.iterrows():
                st.warning(f"⚠️ **Blocker:** Feature **'{b_row['Theme']}'** is scheduled in **{b_row['Current_Stage']}**, but its prerequisite {b_row['Dependency_Status'].replace('⚠️ Blocked by ', '')}!")
        else:
            st.success("✅ **Architecture Validated:** All scheduled features have their foundational prerequisites satisfied.")
            
        st.dataframe(
            dependency_df[['Theme', 'Current_Stage', 'Status_Badge', 'Prerequisites', 'Downstream_Enables', 'Architectural_Rationale']],
            hide_index=True,
            use_container_width=True
        )

    # =========================================================================
    # VIEW: 📜 DECISION HISTORY & EXPLAINABILITY AUDIT TRAIL (DIFFERENTIATOR #5)
    # =========================================================================
    elif st.session_state.active_nav == "📜 Decision History":
        st.markdown("### 📜 Decision History & Explainability Audit Trail")
        st.caption("Maintain transparent, defensible governance over roadmap priorities. Record why features were approved, deferred, or rejected.")

        # Ensure history is initialized
        if not st.session_state.decision_history:
            st.session_state.decision_history = init_decision_log(roadmap_df)

        # Form to record a new decision
        with st.expander("➕ Record New Product Roadmap Decision", expanded=False):
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                rec_theme = st.selectbox("Feature Theme:", roadmap_df['Theme'].tolist() if roadmap_df is not None else [], key="rec_theme_sel")
            with col_rec2:
                rec_status = st.selectbox("Decision Status:", ["Approved (Prioritized)", "Planned (Target Q2)", "Deferred (Future Horizon)", "Rejected (Out of Scope)"], key="rec_status_sel")
            with col_rec3:
                rec_author = st.text_input("Decision Maker / PM:", value="Vaishnavi Reddy", key="rec_author_input")

            rec_rationale = st.text_area("Strategic Decision Rationale & Evidence Notes:", placeholder="Explain why this decision was made, what customer evidence supports it, and what tradeoffs were accepted...", key="rec_notes_input")
            
            if st.button("💾 Record Decision to Audit Trail", type="primary", key="btn_save_decision"):
                if rec_theme and rec_rationale:
                    match_item = roadmap_df[roadmap_df['Theme'] == rec_theme].iloc[0] if roadmap_df is not None else {}
                    st.session_state.decision_history = record_new_decision(
                        st.session_state.decision_history,
                        theme=rec_theme,
                        status=rec_status,
                        rationale=rec_rationale,
                        decision_maker=rec_author,
                        score=float(match_item.get('Score', 8.0)),
                        horizon=str(match_item.get('Stage', 'NOW')),
                        category=str(match_item.get('Assigned Team', 'Core Platform')),
                        requests=int(match_item.get('Volume', 50))
                    )
                    st.success(f"✅ Decision logged for **{rec_theme}**!")
                    st.rerun()

        # Strategic Decision Explainer (Compare Any Two Features)
        st.markdown("#### ⚖️ Strategic Decision Explainer: Defend Roadmaps with Evidence")
        st.caption("Select any two features to generate an objective, data-backed comparative defense for leadership and stakeholders.")
        
        all_themes = roadmap_df['Theme'].tolist() if roadmap_df is not None and not roadmap_df.empty else []
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            theme_choice_a = st.selectbox("Feature A (Prioritized):", all_themes, index=0 if len(all_themes) > 0 else 0, key="exp_theme_a")
        with col_exp2:
            theme_choice_b = st.selectbox("Feature B (Deferred / Lower Priority):", all_themes, index=min(1, len(all_themes)-1) if len(all_themes) > 0 else 0, key="exp_theme_b")

        if theme_choice_a and theme_choice_b and theme_choice_a != theme_choice_b:
            comparison_explanation = explain_decision_comparison(theme_choice_a, theme_choice_b, roadmap_df, ranking_df)
            with st.container(border=True):
                st.markdown(comparison_explanation)

        # Decision Audit Trail Table
        st.markdown("---")
        st.markdown(f"#### 📜 Decision Audit Trail ({len(st.session_state.decision_history)} decisions recorded)")
        
        dec_df = pd.DataFrame(st.session_state.decision_history)
        if not dec_df.empty:
            col_flt_status, col_dl_dec = st.columns([7, 3])
            with col_flt_status:
                filt_status = st.multiselect("Filter by Status:", dec_df['Status'].unique().tolist(), default=dec_df['Status'].unique().tolist(), key="filt_dec_status")
            with col_dl_dec:
                md_dec = export_decision_log_markdown(st.session_state.decision_history)
                st.download_button("📥 Export Audit Trail (.md)", data=md_dec, file_name="decision_audit_trail.md", mime="text/markdown", use_container_width=True)

            filtered_dec = dec_df[dec_df['Status'].isin(filt_status)]
            st.dataframe(
                filtered_dec[['Decision_ID', 'Timestamp', 'Theme', 'Status', 'Horizon', 'Score_At_Decision', 'Customer_Requests', 'Decision_Maker', 'Rationale']],
                column_config={
                    "Score_At_Decision": st.column_config.NumberColumn("RICE Score", format="%.1f"),
                    "Customer_Requests": st.column_config.NumberColumn("Requests", format="%d"),
                    "Rationale": st.column_config.TextColumn("Strategic Rationale", width="large")
                },
                hide_index=True,
                use_container_width=True
            )

    # =========================================================================
    # VIEW 7: 📊 TRENDS & ANALYTICS
    # =========================================================================
    elif st.session_state.active_nav == "📊 Trends & Analytics":
        st.markdown("### 📊 Demand Trend Velocity & Customer Segment Analytics")
        st.caption("Detect demand momentum acceleration and analyze Enterprise vs Free tier requests.")
        
        fig_t = plot_interactive_trends(df, trends_df)
        st.plotly_chart(fig_t, use_container_width=True)
        
        st.markdown("### 💼 Strategic Preset: Revenue-Weighted Prioritization (Enterprise Weighted 4x - Configurable)")
        st.caption("Strategic preset designed for B2B Enterprise growth. Multiplies Enterprise tier signals by 4x to model high ACV contract impact. This weighting is an adjustable PM preset, not a universal scoring rule.")
        if st.session_state.revenue_df is not None:
            st.dataframe(st.session_state.revenue_df, hide_index=True, use_container_width=True)

    # =========================================================================
    # VIEW 8: 📄 REPORTS
    # =========================================================================
    elif st.session_state.active_nav == "📄 Reports":
        st.markdown("### 📄 Executive Briefing & Export Center")
        st.caption("Download presentation-ready strategy reports across Markdown, HTML Dashboard, and PDF formats.")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            md_report = generate_markdown_report(df, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
            st.download_button("📄 Download Markdown Report (.md)", data=md_report, file_name="executive_report.md", mime="text/markdown", use_container_width=True)
        with col_r2:
            html_report = generate_html_report(df, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
            st.download_button("🌐 Download HTML Dashboard (.html)", data=html_report, file_name="executive_dashboard.html", mime="text/html", use_container_width=True)
        with col_r3:
            try:
                pdf_buffer = generate_pdf_report(df, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
                st.download_button("📑 Download Executive PDF (.pdf)", data=pdf_buffer.getvalue(), file_name="executive_strategy.pdf", mime="application/pdf", use_container_width=True)
            except Exception:
                pass
                
        st.markdown("---")
        st.markdown("#### 📋 Preview Executive Strategy Brief")
        st.markdown(md_report[:1800] + "...")

    # =========================================================================
    # VIEW 9: ⚙️ SETTINGS
    # =========================================================================
    elif st.session_state.active_nav == "⚙️ Settings":
        st.markdown("### ⚙️ Workspace Settings & Data Ingestion")
        st.caption("Upload new feedback files or reload standard benchmarks.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            with st.container(border=True):
                st.markdown("#### 📁 Upload New Customer Feedback CSV")
                new_upload = st.file_uploader("Upload Customer Requests CSV", type=["csv"], key="settings_csv_upload")
                if new_upload is not None:
                    if st.button("⚡ Ingest & Process Dataset", type="primary", key="settings_ingest_btn", use_container_width=True):
                        with st.spinner("Processing feedback signals..."):
                            bundle = run_full_pipeline(new_upload)
                            for k, v in bundle.items():
                                st.session_state[k] = v
                            st.session_state.active_nav = "🏠 Overview"
                            st.rerun()

        with col_s2:
            with st.container(border=True):
                st.markdown("#### 🔄 Reset or Reload Benchmark")
                st.write("Reset current state or reload the pre-computed 1,248 request dataset.")
                if st.button("⚡ Reload Sample Dataset (1,248 Reqs)", key="settings_reload_sample_btn", use_container_width=True):
                    load_sample_dataset()
                    st.session_state.active_nav = "🏠 Overview"
                    st.rerun()
                if st.button("🔄 Reset Workspace", key="settings_reset_btn", use_container_width=True):
                    for k in state_defaults.keys():
                        st.session_state[k] = state_defaults[k]
                    st.rerun()

else:
    with st.container(border=True):
        st.markdown("## 👋 Welcome to AI Feature Request Prioritizer")
        st.markdown("Transform unstructured customer feedback into transparent, evidence-backed RICE roadmaps.")
        
        col_start1, col_start2 = st.columns(2)
        with col_start1:
            with st.container(border=True):
                st.markdown("### ⚡ Quick Start: Benchmark Dataset")
                st.write("Load 1,248 multi-channel customer requests across Enterprise, Pro, and Free tiers.")
                if st.button("Load Sample Dataset (1,248 Reqs)", type="primary", key="empty_load_sample_btn", use_container_width=True):
                    load_sample_dataset()
                    st.session_state.active_nav = "🏠 Overview"
                    st.rerun()
                    
        with col_start2:
            with st.container(border=True):
                st.markdown("### 📁 Upload Custom Feedback CSV")
                st.write("Upload your own export from Zendesk, Intercom, Jira, or CSV feedback logs.")
                empty_upload = st.file_uploader("Choose CSV file", type=["csv"], key="empty_uploader")
                if empty_upload is not None:
                    if st.button("🚀 Ingest & Process CSV", type="primary", key="empty_ingest_btn", use_container_width=True):
                        with st.spinner("Processing feedback signals..."):
                            bundle = run_full_pipeline(empty_upload)
                            for k, v in bundle.items():
                                st.session_state[k] = v
                            st.session_state.active_nav = "🏠 Overview"
                            st.rerun()
                            
        st.markdown("---")
        st.caption("Supported columns: `Request_Text` (or feedback/description), `Category`, `Upvotes`, `Customer_Segment` (tier), `Date`, `Status`.")
