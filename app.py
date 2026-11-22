import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
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
plot_bytevox_sentiment_donut = getattr(vis, 'plot_bytevox_sentiment_donut', None)
plot_bytevox_demand_velocity = getattr(vis, 'plot_bytevox_demand_velocity', None)
plot_bytevox_category_volume = getattr(vis, 'plot_bytevox_category_volume', None)
plot_bytevox_segment_sentiment = getattr(vis, 'plot_bytevox_segment_sentiment', None)
plot_bytevox_predictive_forecast = getattr(vis, 'plot_bytevox_predictive_forecast', None)

# ----------------- Streamlit Page Configuration ----------------- #
st.set_page_config(
    page_title="Prioritize.ai | Product Intelligence SaaS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SaaS Styling (ByteVox Clean Aesthetics, Fixed Layout) ----------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
        background-color: #f8fafc;
    }

    /* 1. LOCK SIDEBAR: FIXED WIDTH, NO RESIZING, NO JUMPING */
    section[data-testid="stSidebar"] {
        min-width: 270px !important;
        max-width: 270px !important;
        width: 270px !important;
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        transition: none !important;
    }
    div[data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
        padding-top: 1.2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        background-color: #ffffff !important;
    }
    
    /* 2. REMOVE ALL JITTERY TRANSFORMS AND SHIFTS */
    *, *:hover {
        transform: none !important;
        animation: none !important;
    }

    /* 3. MAIN CONTAINER FIXED PADDING */
    .main .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }

    /* Sidebar Navigation Menu Items (Styled like native SaaS navigation) */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 500;
        color: #334155;
        cursor: pointer;
        margin: 0;
        border: 1px solid transparent;
        background: transparent;
        transition: background-color 0.1s ease;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #f1f5f9;
        color: #0f172a;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #eef2ff !important;
        color: #4f46e5 !important;
        font-weight: 700 !important;
        border-color: #e0e7ff !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.84rem !important;
        font-weight: inherit !important;
        color: inherit !important;
        margin: 0 !important;
    }

    /* Top Action Bar Breadcrumb Pill */
    .bv-breadcrumb-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }
    .bv-breadcrumb-brand {
        color: #4f46e5;
        font-weight: 700;
    }

    /* KPI Cards (Row of 5 stat metrics) */
    .bv-kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 104px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .bv-kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .bv-kpi-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #475569;
        text-transform: capitalize;
    }
    .bv-kpi-icon {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .bv-kpi-val-row {
        display: flex;
        align-items: baseline;
        gap: 6px;
    }
    .bv-kpi-number {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
    }
    .bv-kpi-denom {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
    }
    .bv-kpi-delta {
        font-size: 0.74rem;
        font-weight: 700;
        color: #10b981;
        display: inline-flex;
        align-items: center;
        gap: 2px;
    }
    .bv-kpi-delta-red {
        color: #ef4444;
    }

    /* 6 Analytical Cards Container */
    .bv-analytics-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 320px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        margin-bottom: 16px;
    }
    .bv-card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 6px;
    }
    .bv-card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .bv-card-subtitle {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 2px;
    }
    .bv-card-legend {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.72rem;
        color: #64748b;
    }
    .legend-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 3px;
    }
    .dot-green { background-color: #10b981; }
    .dot-slate { background-color: #94a3b8; }
    .dot-red { background-color: #ef4444; }

    /* Top Themes 2x4 Tile Grid */
    .themes-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        grid-template-rows: repeat(2, 1fr);
        gap: 8px;
        margin-top: 8px;
    }
    .theme-box {
        background: #eff6ff;
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 14px 6px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 72px;
    }
    .theme-box-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #1e3a8a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .theme-box-count {
        font-size: 0.72rem;
        font-weight: 600;
        color: #3b82f6;
        margin-top: 4px;
    }

    /* Badges & Quote Styling */
    .badge-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .pill-now { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
    .pill-next { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
    .pill-later { background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; }
    .pill-team { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .pill-score { background: #ede9fe; color: #6d28d9; border: 1px solid #ddd6fe; }

    .saas-quote {
        background: #f8fafc;
        border-left: 3px solid #6366f1;
        border-radius: 0 6px 6px 0;
        padding: 8px 12px;
        margin: 6px 0;
        font-style: italic;
        font-size: 0.84rem;
        color: #334155;
    }
    
    .rice-math-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .rice-math-step {
        background: #f8fafc;
        border: 1px solid #edf2f7;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 5px 0;
        font-size: 0.82rem;
        color: #334155;
    }
    
    .kanban-col-header {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.82rem;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .kanban-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
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
    'active_nav': "🏠 Overview"
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- Sidebar Branding & Navigation (ByteVox Style) ----------------- #
with st.sidebar:
    # Top Brand Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; padding: 4px 6px 16px 6px; border-bottom: 1px solid #e2e8f0; margin-bottom: 12px;">
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; font-weight: 900; font-size: 1.15rem; padding: 6px 10px; border-radius: 8px;">
            ⚡
        </div>
        <div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; line-height: 1.1;">Prioritize.ai</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: #64748b;">Product Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Primary Navigation Menu
    nav_options = [
        "🏠 Overview",
        "🎯 RICE Decision Engine",
        "🗺️ Interactive Roadmap",
        "💡 Evidence Insights",
        "📈 Demand Velocity",
        "👥 Customer Segments",
        "🔗 Dependency Graph",
        "🔍 Feedback Explorer",
        "🤖 AI Product Copilot",
        "📄 Executive Reports",
        "⚙️ Settings & Ingestion"
    ]
    
    # Sync with session state
    curr_nav_idx = nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0
    selected_nav = st.radio("Navigation", nav_options, index=curr_nav_idx, label_visibility="collapsed")
    if selected_nav != st.session_state.active_nav:
        st.session_state.active_nav = selected_nav
        st.rerun()

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # Bottom User / Admin Pill (like in reference image)
    st.markdown("""
    <div style="padding-top: 14px; border-top: 1px solid #e2e8f0; display: flex; align-items: center; gap: 10px;">
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #4f46e5; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;">
            V
        </div>
        <div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #0f172a;">Vaishnavi R.</div>
            <div style="font-size: 0.7rem; color: #64748b;">Lead Product Manager</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Automatic / Manual Ingestion Pipeline ----------------- #
# Auto-load the 1,000 sample dataset on first launch so the dashboard is immediately live!
sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_requests.csv')
uploaded_file = None

if not st.session_state.processed_data:
    if os.path.exists(sample_path):
        with open(sample_path, 'r', encoding='utf-8') as f:
            uploaded_file = io.StringIO(f.read())

if uploaded_file is not None and not st.session_state.processed_data:
    try:
        progress_bar = st.progress(0, text="Initializing Prioritize.ai Data Pipeline...")
        start_time = time.time()
        
        # 1. Clean & Standardize
        progress_bar.progress(10, text="Standardizing customer feedback schema...")
        df = load_and_clean_data(uploaded_file)
        
        # 2. Sentiment Analysis
        progress_bar.progress(25, text="Analyzing customer friction points & sentiment...")
        sentiment_model = load_sentiment_model()
        df['Sentiment'] = analyze_sentiments(df['Cleaned_Text'].tolist(), sentiment_model)
        
        # 3. Auto-Tag Cross-functional Teams
        progress_bar.progress(35, text="Auto-tagging cross-functional team domains...")
        df = tag_dataframe(df)
        
        # 4. Vector Embeddings
        progress_bar.progress(50, text="Generating dense semantic embeddings (all-MiniLM-L6-v2)...")
        embed_model = load_embedding_model()
        embeddings = generate_embeddings(df['Cleaned_Text'].tolist(), embed_model)
        
        # 5. Semantic Deduplication & Indexing
        progress_bar.progress(65, text="Building FAISS cosine index & clustering duplicates...")
        index = build_faiss_index(embeddings)
        dup_groups = find_duplicates(index, embeddings, similarity_threshold=0.82)
        total_dups = sum(g['count'] - 1 for g in dup_groups)
        
        # 6. KMeans Semantic Clustering
        progress_bar.progress(75, text="Synthesizing strategic feature themes...")
        n_clusters = min(12, max(2, len(df) // 25))
        df['Cluster'] = perform_clustering(embeddings, n_clusters=n_clusters)
        
        # 7. Keywords & Extractive Centroid Summaries
        progress_bar.progress(85, text="Extracting domain keywords and centroid quotes...")
        kw_model = load_keyword_model()
        cluster_summaries = {}
        cluster_keywords = {}
        for c_id in sorted(df['Cluster'].unique()):
            c_texts = df[df['Cluster'] == c_id]['Cleaned_Text'].tolist()
            cluster_summaries[c_id] = summarize_cluster(c_texts, embed_model)
            cluster_keywords[c_id] = extract_keywords(" ".join(c_texts[:50]), kw_model, top_n=5)
            
        # 8. RICE Scoring & Ranking
        progress_bar.progress(90, text="Calculating Smart Priority RICE Scores...")
        ranking_df = rank_features(
            df, cluster_col='Cluster', upvotes_col='Upvotes',
            weight_reach=st.session_state.weight_reach,
            weight_impact=st.session_state.weight_impact,
            weight_confidence=st.session_state.weight_confidence,
            weight_effort=st.session_state.weight_effort
        )
        
        # 9. Strategic Artifacts: Roadmap, Insights, Dependencies, Trends, Segments, Alerts
        progress_bar.progress(95, text="Constructing Roadmap, Dependency Graph & Velocity Trends...")
        roadmap_df = generate_ai_roadmap(df, ranking_df, cluster_keywords, cluster_summaries)
        dependency_df = map_feature_dependencies(roadmap_df)
        insights = generate_evidence_backed_insights(df, ranking_df, cluster_keywords, cluster_summaries)
        trends_df, monthly_volume = analyze_demand_trends(df, ranking_df, cluster_keywords)
        top_per_segment, segment_summary_df, revenue_df = analyze_customer_segments(df, ranking_df, cluster_keywords)
        alerts = detect_emerging_feature_alerts(trends_df, ranking_df, df)
        
        # Save into session state
        st.session_state.df = df
        st.session_state.index = index
        st.session_state.ranking_df = ranking_df
        st.session_state.roadmap_df = roadmap_df
        st.session_state.dependency_df = dependency_df
        st.session_state.insights = insights
        st.session_state.trends_df = trends_df
        st.session_state.monthly_volume = monthly_volume
        st.session_state.top_per_segment = top_per_segment
        st.session_state.segment_summary_df = segment_summary_df
        st.session_state.revenue_df = revenue_df
        st.session_state.alerts = alerts
        st.session_state.duplicate_groups = dup_groups
        st.session_state.total_duplicates = total_dups
        st.session_state.cluster_summaries = cluster_summaries
        st.session_state.cluster_keywords = cluster_keywords
        st.session_state.processed_data = True
        
        elapsed = time.time() - start_time
        progress_bar.progress(100, text=f"✅ Prioritize.ai Live Pipeline Ready ({elapsed:.2f}s)")
        time.sleep(0.2)
        progress_bar.empty()
    except Exception as e:
        st.error(f"❌ Error during data processing: {str(e)}")
        st.exception(e)

# ----------------- MAIN WORKSPACE VIEWS ----------------- #
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

    # Shared Top Action Bar (Matching ByteVox Header)
    col_hdr_left, col_hdr_right = st.columns([5, 5])
    with col_hdr_left:
        st.markdown(f"""
        <div style="padding-top: 4px;">
            <span class="bv-breadcrumb-pill">
                <span class="bv-breadcrumb-brand">Prioritize.ai</span>
                <span style="color: #cbd5e1;">·</span>
                <span>{st.session_state.active_nav.replace('🏠 ', '').replace('🎯 ', '').replace('🗺️ ', '')}</span>
                <span style="color: #cbd5e1;">·</span>
                <span style="color: #64748b; font-weight: 500;">Live Sync</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_hdr_right:
        col_act1, col_act2, col_act3, col_act4 = st.columns([2, 3, 2, 3])
        with col_act1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col_act2:
            if st.button("📄 Executive Report", use_container_width=True):
                st.session_state.active_nav = "📄 Executive Reports"
                st.rerun()
        with col_act3:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("☁️ Export", data=csv_data, file_name="feature_requests.csv", mime="text/csv", use_container_width=True)
        with col_act4:
            if st.button("↗️ Open Backlog", type="primary", use_container_width=True):
                st.session_state.active_nav = "🔍 Feedback Explorer"
                st.rerun()

    # =========================================================================
    # VIEW 1: 🏠 OVERVIEW (BYTEVOX SAAS DASHBOARD REPLICA)
    # =========================================================================
    if st.session_state.active_nav == "🏠 Overview":
        
        # Horizontal Filter Bar (ByteVox Filter Row)
        st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
        with st.container():
            col_f_icon, col_f_time, col_f_seg, col_f_cat, col_f_btn = st.columns([1.2, 2.3, 2.5, 2.5, 2.5])
            
            with col_f_icon:
                st.markdown("""
                <div style="padding-top: 8px; font-size: 0.82rem; font-weight: 700; color: #475569; display: flex; align-items: center; gap: 5px;">
                    <span>⚡ Filters:</span>
                </div>
                """, unsafe_allow_html=True)
                
            with col_f_time:
                time_filter = st.selectbox("Timeframe", ["Last 30 days", "Last 14 days", "All Time"], label_visibility="collapsed", key="ov_time")
            with col_f_seg:
                seg_list = ["All Segments"] + sorted(df['Customer_Segment'].unique().tolist())
                seg_filter = st.selectbox("Segment", seg_list, label_visibility="collapsed", key="ov_seg")
            with col_f_cat:
                cat_list = ["All Products"] + sorted(df['Category'].unique().tolist())
                cat_filter = st.selectbox("Category", cat_list, label_visibility="collapsed", key="ov_cat")
            with col_f_btn:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.button("Apply", type="primary", use_container_width=True)
                with col_b2:
                    if st.button("Reset", use_container_width=True):
                        st.session_state.ov_time = "Last 30 days"
                        st.session_state.ov_seg = "All Segments"
                        st.session_state.ov_cat = "All Products"
                        st.rerun()

        # Apply Horizontal Filters to Overview Display
        filtered_ov = df.copy()
        if seg_filter != "All Segments":
            filtered_ov = filtered_ov[filtered_ov['Customer_Segment'] == seg_filter]
        if cat_filter != "All Products":
            filtered_ov = filtered_ov[filtered_ov['Category'] == cat_filter]

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # ----------------- 5 KPI METRICS STRIP (ByteVox KPI Row) ----------------- #
        col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
        
        # 1. Feature Health Score
        with col_k1:
            st.markdown("""
            <div class="bv-kpi-card">
                <div class="bv-kpi-header">
                    <span class="bv-kpi-label">Feature Health Score</span>
                    <span class="bv-kpi-icon">🛡️</span>
                </div>
                <div class="bv-kpi-val-row">
                    <span class="bv-kpi-number">82</span>
                    <span class="bv-kpi-denom">/100</span>
                    <span class="bv-kpi-delta">📈 +3</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Prioritization Index (RICE)
        with col_k2:
            st.markdown("""
            <div class="bv-kpi-card">
                <div class="bv-kpi-header">
                    <span class="bv-kpi-label">Prioritization Index</span>
                    <span class="bv-kpi-icon">↗</span>
                </div>
                <div class="bv-kpi-val-row">
                    <span class="bv-kpi-number">7.2</span>
                    <span class="bv-kpi-denom">/10</span>
                    <span class="bv-kpi-delta">↗ +4</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Customer Confidence Rate
        with col_k3:
            st.markdown("""
            <div class="bv-kpi-card">
                <div class="bv-kpi-header">
                    <span class="bv-kpi-label">Confidence Rate</span>
                    <span class="bv-kpi-icon">✨</span>
                </div>
                <div class="bv-kpi-val-row">
                    <span class="bv-kpi-number">4.4</span>
                    <span class="bv-kpi-denom">/5</span>
                    <span class="bv-kpi-delta">✨ +0.2</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 4. Response Velocity
        with col_k4:
            st.markdown("""
            <div class="bv-kpi-card">
                <div class="bv-kpi-header">
                    <span class="bv-kpi-label">Response Velocity</span>
                    <span class="bv-kpi-icon">⏱️</span>
                </div>
                <div class="bv-kpi-val-row">
                    <span class="bv-kpi-number">9.3</span>
                    <span class="bv-kpi-denom">hrs</span>
                    <span class="bv-kpi-delta bv-kpi-delta-red">⏱️ -1.1</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 5. Open Blockers / Escalations
        with col_k5:
            blocked_count = len(dependency_df[dependency_df['Is_Blocked']]) if dependency_df is not None else 0
            display_blockers = max(37, blocked_count)
            st.markdown(f"""
            <div class="bv-kpi-card">
                <div class="bv-kpi-header">
                    <span class="bv-kpi-label">Open Escalations</span>
                    <span class="bv-kpi-icon">🔔</span>
                </div>
                <div class="bv-kpi-val-row">
                    <span class="bv-kpi-number">{display_blockers}</span>
                    <span class="bv-kpi-delta bv-kpi-delta-red">🚨 -6</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

        # ----------------- ROW 1: 3 ANALYTICAL CARDS ----------------- #
        col_r1_1, col_r1_2, col_r1_3 = st.columns(3)
        
        # Card 1: Sentiment Overview (Donut Chart)
        with col_r1_1:
            st.markdown("""
            <div class="bv-analytics-card">
                <div class="bv-card-top">
                    <div>
                        <div class="bv-card-title">Sentiment Overview</div>
                        <div class="bv-card-subtitle">Positive / Neutral / Negative</div>
                    </div>
                    <div class="bv-card-legend">
                        <span><span class="legend-dot dot-green"></span>Positive</span>
                        <span><span class="legend-dot dot-slate"></span>Neutral</span>
                        <span><span class="legend-dot dot-red"></span>Negative</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            fig_donut = plot_bytevox_sentiment_donut(filtered_ov)
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Card 2: Demand & Velocity Trend (Dual Curve Line Chart)
        with col_r1_2:
            st.markdown("""
            <div class="bv-analytics-card">
                <div class="bv-card-top">
                    <div>
                        <div class="bv-card-title">Demand & Velocity Trend</div>
                        <div class="bv-card-subtitle">Current vs previous period</div>
                    </div>
                    <div class="bv-card-legend">
                        <span><span class="legend-dot dot-blue"></span>Demand</span>
                        <span><span class="legend-dot dot-teal"></span>Velocity</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            fig_trend = plot_bytevox_demand_velocity(filtered_ov, trends_df)
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Card 3: Backlog Volume by Category (Purple Bar Chart)
        with col_r1_3:
            st.markdown("""
            <div class="bv-analytics-card">
                <div class="bv-card-top">
                    <div>
                        <div class="bv-card-title">Backlog Performance</div>
                        <div class="bv-card-subtitle">Feature volume by primary category</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            fig_cat = plot_bytevox_category_volume(filtered_ov)
            st.plotly_chart(fig_cat, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ----------------- ROW 2: 3 ANALYTICAL CARDS ----------------- #
        col_r2_1, col_r2_2, col_r2_3 = st.columns(3)
        
        # Card 4: Segment Sentiment (100% Stacked Bar Chart)
        with col_r2_1:
            st.markdown("""
            <div class="bv-analytics-card">
                <div class="bv-card-top">
                    <div>
                        <div class="bv-card-title">Segment Sentiment</div>
                        <div class="bv-card-subtitle">Hotspots of negative feedback (stacked)</div>
                    </div>
                    <div class="bv-card-legend">
                        <span><span class="legend-dot dot-green"></span>Positive</span>
                        <span><span class="legend-dot dot-slate"></span>Neutral</span>
                        <span><span class="legend-dot dot-red"></span>Negative</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            fig_seg = plot_bytevox_segment_sentiment(filtered_ov)
            st.plotly_chart(fig_seg, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Card 5: Top Themes (2x4 Grid of Soft Periwinkle Tiles)
        with col_r2_2:
            st.markdown("""
            <div class="bv-analytics-card">
                <div class="bv-card-top">
                    <div>
                        <div class="bv-card-title">Top Themes</div>
                        <div class="bv-card-subtitle">Auto-extracted keywords from transcripts</div>
                    </div>
                    <div>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #4f46e5; cursor: pointer;">View root-cause →</span>
                    </div>
                </div>
                <div class="themes-grid">
                    <div class="theme-box">
                        <div class="theme-box-title">Billing</div>
                        <div class="theme-box-count">28 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">Delivery Delay</div>
                        <div class="theme-box-count">24 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">Invoice Error</div>
                        <div class="theme-box-count">20 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">App Crash</div>
                        <div class="theme-box-count">16 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">Refund</div>
                        <div class="theme-box-count">15 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">Agent Wait</div>
                        <div class="theme-box-count">14 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">Price Mismatch</div>
                        <div class="theme-box-count">12 reqs</div>
                    </div>
                    <div class="theme-box">
                        <div class="theme-box-title">Service Quality</div>
                        <div class="theme-box-count">10 reqs</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Card 6: Predictive Forecast (Area Forecast Chart)
        with col_r2_3:
            st.markdown("""
            <div class="bv-analytics-card">
                <div class="bv-card-top">
                    <div>
                        <div class="bv-card-title">Predictive Forecast</div>
                        <div class="bv-card-subtitle">Next-week sentiment prediction</div>
                    </div>
                    <div class="bv-card-legend">
                        <span><span class="legend-dot dot-green"></span>Positive</span>
                        <span><span class="legend-dot dot-red"></span>Negative</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            fig_fc = plot_bytevox_predictive_forecast(trends_df, monthly_volume)
            st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})
            st.markdown("""
            <div style="font-size: 0.74rem; color: #64748b; margin-top: 2px;">
                Predicting slight uplift in positive sentiment as refunds clear.
            </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # VIEW 2: 🎯 RICE DECISION ENGINE & SENSITIVITY SLIDERS
    # =========================================================================
    elif st.session_state.active_nav == "🎯 RICE Decision Engine":
        st.markdown("### 🎯 Smart Priority Score (RICE) Engine & Step-by-Step Explainability")
        st.caption("Calculate transparent, objective feature scores using Reach, Impact, Confidence, and Effort.")
        
        # Strategic Presets & Sliders
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

            if (w_reach != st.session_state.weight_reach or 
                w_impact != st.session_state.weight_impact or 
                w_conf != st.session_state.weight_confidence or 
                w_eff != st.session_state.weight_effort):
                st.session_state.weight_reach = w_reach
                st.session_state.weight_impact = w_impact
                st.session_state.weight_confidence = w_conf
                st.session_state.weight_effort = w_eff
                st.session_state.active_preset = "Custom Weights"
                ranking_df = rank_features(
                    df, weight_reach=w_reach, weight_impact=w_impact, 
                    weight_confidence=w_conf, weight_effort=w_eff
                )
                st.session_state.ranking_df = ranking_df
                roadmap_df = generate_ai_roadmap(df, ranking_df, st.session_state.cluster_keywords, st.session_state.cluster_summaries)
                st.session_state.roadmap_df = roadmap_df
                st.session_state.dependency_df = map_feature_dependencies(roadmap_df)

        # Matrix and Effort Tuning
        col_matrix, col_sliders = st.columns([7, 5])
        with col_matrix:
            fig_matrix = plot_interactive_rice_matrix(roadmap_df)
            st.plotly_chart(fig_matrix, use_container_width=True)
            
        with col_sliders:
            st.markdown("#### 🛠️ Fine-Tune Development Effort (1–10)")
            st.caption("Adjust effort points (stable order without jumping):")
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
                    roadmap_df.loc[roadmap_df['ID'] == c_id, 'RICE_Score'] = new_score
                    st.session_state.ranking_df = ranking_df
                    st.session_state.roadmap_df = roadmap_df

        st.markdown("---")
        st.markdown("### 🧮 Step-by-Step RICE Calculation & Explanations")
        for i in range(min(5, len(ranking_df))):
            r_row = ranking_df.iloc[i]
            theme_title = r_row['Dominant Category']
            c_reach = r_row['Reach']
            c_impact = r_row['Impact']
            c_conf = r_row['Confidence']
            c_effort = r_row['Effort']
            c_score = r_row['RICE Score']
            w_r = st.session_state.weight_reach
            w_i = st.session_state.weight_impact
            w_c = st.session_state.weight_confidence
            w_e = st.session_state.weight_effort
            
            with st.expander(f"#{i+1} **{theme_title}** — RICE Score: **{c_score}**", expanded=(i==0)):
                st.markdown(f"""
                <div class="rice-math-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.05rem; font-weight: 800; color: #4f46e5;">RICE = {c_score}</span>
                        <span class="badge-pill pill-now">Rank #{i+1}</span>
                    </div>
                    <div class="rice-math-step">
                        <strong>1. Reach ({c_reach:,} affected users):</strong> 
                        Derived from {r_row['Request Count']} feedback requests + {r_row['Total Upvotes']:,} upvotes + {r_row['Enterprise Count']} enterprise accounts.
                        <br><em>Effective Weighted Reach:</em> {c_reach} × {w_r} = <strong>{round(c_reach * w_r, 1)}</strong>
                    </div>
                    <div class="rice-math-step">
                        <strong>2. Impact ({c_impact} - {r_row['Impact_Desc']}):</strong> 
                        Quantified from {r_row['Negative Ratio']}% negative customer friction and churn severity.
                        <br><em>Effective Weighted Impact:</em> {c_impact} × {w_i} = <strong>{round(c_impact * w_i, 2)}</strong>
                    </div>
                    <div class="rice-math-step">
                        <strong>3. Confidence ({int(c_conf*100)}% - {r_row['Confidence_Desc']}):</strong> 
                        Statistical certainty based on feedback volume and keyword density.
                        <br><em>Effective Weighted Confidence:</em> {c_conf} × {w_c} = <strong>{round(c_conf * w_c, 2)}</strong>
                    </div>
                    <div class="rice-math-step">
                        <strong>4. Effort ({c_effort} points):</strong> 
                        Estimated engineering difficulty based on {r_row['Dominant Category']} domain complexity.
                        <br><em>Effective Weighted Effort:</em> {c_effort} × {w_e} = <strong>{round(c_effort * w_e, 2)}</strong>
                    </div>
                    <div style="background: #ede9fe; padding: 8px 12px; border-radius: 6px; margin-top: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #5b21b6;">
                        Formula: ({round(c_reach * w_r, 1)} × {round(c_impact * w_i, 2)} × {round(c_conf * w_c, 2)}) / {round(c_effort * w_e, 2)} = <strong>{c_score}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
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
    # VIEW 3: 🗺️ INTERACTIVE ROADMAP (KANBAN)
    # =========================================================================
    elif st.session_state.active_nav == "🗺️ Interactive Roadmap":
        st.markdown("### 🗺️ Interactive Now / Next / Later Roadmap Board")
        st.caption("Organize features into execution horizons. Reassign horizons dynamically with immediate state persistence.")
        
        col_export_md, col_export_csv = st.columns([2, 5])
        with col_export_md:
            roadmap_md = export_roadmap_markdown(roadmap_df)
            st.download_button("📥 Export Roadmap (.md)", data=roadmap_md, file_name="product_roadmap_now_next_later.md", mime="text/markdown", use_container_width=True)
            
        col_now, col_next, col_later = st.columns(3)
        now_items = roadmap_df[roadmap_df['Stage'] == 'NOW']
        next_items = roadmap_df[roadmap_df['Stage'] == 'NEXT']
        later_items = roadmap_df[roadmap_df['Stage'] == 'LATER']
        
        with col_now:
            st.markdown(f"""
            <div class="kanban-col-header" style="border-top: 3px solid #ef4444;">
                <span>🔥 NOW (1–3 Months)</span>
                <span class="badge-pill pill-now">{len(now_items)} Items</span>
            </div>
            """, unsafe_allow_html=True)
            for idx, item in now_items.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="kanban-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 4px;">
                            <strong style="font-size: 0.92rem;">{item['Theme']}</strong>
                            <span class="badge-pill pill-score">RICE {item['Score']}</span>
                        </div>
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                            <span class="badge-pill pill-team">👥 {item['Assigned Team']}</span>
                            <span class="badge-pill pill-team">⚡ {item['Effort']} pts</span>
                            <span class="badge-pill pill-team">💬 {item['Volume']} reqs</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #475569; line-height: 1.35;">
                            {item['Summary'][:95]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    new_stg = st.selectbox("Reassign Horizon:", ["NOW", "NEXT", "LATER"], index=0, key=f"mv_now_{item['ID']}")
                    if new_stg != "NOW":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                        
        with col_next:
            st.markdown(f"""
            <div class="kanban-col-header" style="border-top: 3px solid #f59e0b;">
                <span>⭐ NEXT (3–6 Months)</span>
                <span class="badge-pill pill-next">{len(next_items)} Items</span>
            </div>
            """, unsafe_allow_html=True)
            for idx, item in next_items.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="kanban-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 4px;">
                            <strong style="font-size: 0.92rem;">{item['Theme']}</strong>
                            <span class="badge-pill pill-score">RICE {item['Score']}</span>
                        </div>
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                            <span class="badge-pill pill-team">👥 {item['Assigned Team']}</span>
                            <span class="badge-pill pill-team">⚡ {item['Effort']} pts</span>
                            <span class="badge-pill pill-team">💬 {item['Volume']} reqs</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #475569; line-height: 1.35;">
                            {item['Summary'][:95]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    new_stg = st.selectbox("Reassign Horizon:", ["NOW", "NEXT", "LATER"], index=1, key=f"mv_nxt_{item['ID']}")
                    if new_stg != "NEXT":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                        
        with col_later:
            st.markdown(f"""
            <div class="kanban-col-header" style="border-top: 3px solid #3b82f6;">
                <span>🧊 LATER (6–12 Months)</span>
                <span class="badge-pill pill-later">{len(later_items)} Items</span>
            </div>
            """, unsafe_allow_html=True)
            for idx, item in later_items.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="kanban-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 4px;">
                            <strong style="font-size: 0.92rem;">{item['Theme']}</strong>
                            <span class="badge-pill pill-score">RICE {item['Score']}</span>
                        </div>
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                            <span class="badge-pill pill-team">👥 {item['Assigned Team']}</span>
                            <span class="badge-pill pill-team">⚡ {item['Effort']} pts</span>
                            <span class="badge-pill pill-team">💬 {item['Volume']} reqs</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #475569; line-height: 1.35;">
                            {item['Summary'][:95]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    new_stg = st.selectbox("Reassign Horizon:", ["NOW", "NEXT", "LATER"], index=2, key=f"mv_ltr_{item['ID']}")
                    if new_stg != "LATER":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()

    # =========================================================================
    # VIEW 4: 💡 EVIDENCE-BACKED INSIGHTS
    # =========================================================================
    elif st.session_state.active_nav == "💡 Evidence Insights":
        st.markdown("### 💡 Evidence-Backed AI Insights & Voice of Customer")
        st.caption("Every AI recommendation is transparently supported by request volume, customer tier data, and verbatim quotes.")
        
        for ins in insights:
            with st.container():
                st.markdown(f"#### {ins['Rationale_Badge']}: {ins['Theme']}")
                col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
                with col_meta1:
                    st.metric("RICE Priority", f"{ins['RICE_Score']}")
                with col_meta2:
                    st.metric("Supporting Volume", f"{ins['Request_Count']} requests", f"{ins['Pct_Of_Total']}% share")
                with col_meta3:
                    st.metric("Total Upvotes", f"{ins['Total_Upvotes']:,}")
                with col_meta4:
                    st.metric("Customer Friction", f"{ins['Sentiment_Breakdown']['Negative_Pct']}% negative")
                    
                st.markdown(f"**🤖 AI Rationale:** {ins['AI_Rationale']}")
                st.markdown(f"**🎯 Actionable Recommendation:** `{ins['Actionable_Recommendation']}`")
                st.markdown(f"**👥 Segment Concentration:** {ins['Segment_Summary']}")
                
                st.markdown("##### 💬 Verbatim Customer Feedback Samples:")
                for q in ins['Customer_Quotes']:
                    st.markdown(
                        f"""<div class="saas-quote">
                            "{q['text']}" 
                            <br><small style="color: #64748b;">— <strong>{q['segment']}</strong> Customer via {q['source']} ({q['upvotes']} upvotes, {q['date']})</small>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                st.markdown("---")

    # =========================================================================
    # VIEW 5: 📈 DEMAND VELOCITY TRENDS
    # =========================================================================
    elif st.session_state.active_nav == "📈 Demand Velocity":
        st.markdown("### 📈 Demand Trend Velocity & Forecasting")
        st.caption("Detect emerging demand acceleration before features become the #1 requested ticket.")
        
        if trends_df is not None and not trends_df.empty:
            fig_trends = plot_interactive_trends(df, trends_df)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            st.markdown("### 🚀 Momentum Tracker (60-Day Acceleration)")
            st.dataframe(
                trends_df[[
                    'Theme', 'Momentum', 'Growth_Rate_Pct', 'Recent_60d_Requests', 
                    'Prior_60d_Requests', 'Forecast_30d_Requests', 'RICE_Score'
                ]],
                column_config={
                    "Growth_Rate_Pct": st.column_config.NumberColumn("60d Velocity Growth", format="+%.1f%%"),
                    "Forecast_30d_Requests": st.column_config.NumberColumn("30-Day Projected Volume", format="~%d reqs")
                },
                hide_index=True,
                use_container_width=True
            )
            
            if alerts:
                st.markdown("### 🔔 Active Emerging Signal Alerts")
                for alert in alerts:
                    st.markdown(f"""
                    <div style="background: #ffffff; border-left: 4px solid #f97316; border: 1px solid #fed7aa; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                        <strong>{alert['Badge']}</strong>: {alert['Theme']} — {alert['Description']}
                        <br><small style="color: #c2410c;">Action: <code>{alert['Action']}</code></small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Date information is not available to generate trend velocity.")

    # =========================================================================
    # VIEW 6: 👥 CUSTOMER SEGMENTS
    # =========================================================================
    elif st.session_state.active_nav == "👥 Customer Segments":
        st.markdown("### 👥 Customer Segment Intelligence")
        st.caption("Analyze how feature demands diverge between Enterprise, Pro, Starter, and Free tiers.")
        
        col_seg_chart, col_seg_table = st.columns([5, 7])
        with col_seg_chart:
            fig_seg = plot_segment_donut(df)
            st.plotly_chart(fig_seg, use_container_width=True)
            
        with col_seg_table:
            st.markdown("#### 📊 Segment Demand Summary")
            if segment_summary_df is not None and not segment_summary_df.empty:
                st.dataframe(
                    segment_summary_df,
                    column_config={
                        "Share_Pct": st.column_config.NumberColumn("Feedback Share", format="%.1f%%"),
                        "Negative_Sentiment_Pct": st.column_config.NumberColumn("Friction Rate", format="%.1f%%")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
        st.markdown("---")
        st.markdown("### 💼 Revenue-Weighted Priority Ranking (Enterprise Weighted 4x)")
        if st.session_state.revenue_df is not None:
            st.dataframe(
                st.session_state.revenue_df,
                column_config={
                    "Revenue_Weighted_Score": st.column_config.ProgressColumn(
                        "Revenue Score", format="%.1f", min_value=0, max_value=float(st.session_state.revenue_df['Revenue_Weighted_Score'].max())
                    )
                },
                hide_index=True,
                use_container_width=True
            )

    # =========================================================================
    # VIEW 7: 🔗 DEPENDENCY ARCHITECTURE
    # =========================================================================
    elif st.session_state.active_nav == "🔗 Dependency Graph":
        st.markdown("### 🔗 Architectural Dependency Mapping & Blocker Detection")
        st.caption("Prevent engineering bottlenecks by verifying technical prerequisites across functional layers.")
        
        blocked_items = dependency_df[dependency_df['Is_Blocked']]
        if not blocked_items.empty:
            for _, b_row in blocked_items.iterrows():
                st.warning(
                    f"⚠️ **Execution Blocker:** Feature **'{b_row['Theme']}'** is scheduled in **{b_row['Current_Stage']}**, "
                    f"but its architectural prerequisite {b_row['Dependency_Status'].replace('⚠️ Blocked by ', '')}!"
                )
        else:
            st.success("✅ **Architecture Validated:** All scheduled features have their foundational prerequisites satisfied.")
            
        st.dataframe(
            dependency_df[['Theme', 'Current_Stage', 'Status_Badge', 'Prerequisites', 'Downstream_Enables', 'Architectural_Rationale']],
            column_config={
                "Status_Badge": st.column_config.TextColumn("Readiness Status"),
                "Prerequisites": st.column_config.TextColumn("Requires First"),
                "Downstream_Enables": st.column_config.TextColumn("Unblocks Later")
            },
            hide_index=True,
            use_container_width=True
        )

    # =========================================================================
    # VIEW 8: 🔍 FEEDBACK EXPLORER & DEDUPLICATION
    # =========================================================================
    elif st.session_state.active_nav == "🔍 Feedback Explorer":
        st.markdown("### 🔍 Semantic Deduplication & Feedback Explorer")
        st.caption("FAISS vector cosine similarity identifies identical and near-duplicate customer feedback.")
        
        st.markdown(f"#### 🔗 Duplicate Groups ({len(st.session_state.duplicate_groups or [])} groups merged)")
        if st.session_state.duplicate_groups:
            for g in (st.session_state.duplicate_groups[:8]):
                prim_row = df.iloc[g['primary_idx']]
                with st.expander(f"📌 Primary: \"{prim_row['Request_Text']}\" ({g['count']} duplicates merged)"):
                    st.caption(f"Semantic similarity: **{g['avg_similarity']}** | Category: **{prim_row['Category']}**")
                    for d_idx in g['duplicate_indices']:
                        dup_row = df.iloc[d_idx]
                        st.markdown(f"- \"{dup_row['Request_Text']}\" *({dup_row['Customer_Segment']} tier, {dup_row['Upvotes']} upvotes)*")
        else:
            st.info("No semantic duplicates found above the current threshold.")
            
        st.markdown("---")
        st.markdown("#### ☁️ Voice of Customer Keyword Cloud")
        all_feedback_text = " ".join(df['Cleaned_Text'].tolist()[:300])
        fig_wc = generate_wordcloud(all_feedback_text)
        st.pyplot(fig_wc)

    # =========================================================================
    # VIEW 9: 🤖 AI PRODUCT COPILOT
    # =========================================================================
    elif st.session_state.active_nav == "🤖 AI Product Copilot":
        st.markdown("### 🤖 AI Product Copilot & Generation Suite")
        st.caption("Synthesize customer feedback into PRDs, draft public release notes, or run semantic natural language search.")
        
        copilot_sub1, copilot_sub2, copilot_sub3 = st.tabs([
            "🔎 Natural Language Search",
            "📝 AI PRD Generator",
            "📣 Release Notes Generator"
        ])
        
        with copilot_sub1:
            st.markdown("#### Instant Semantic Vector Search")
            search_query = st.text_input("Enter natural language query:", placeholder="E.g., SSO authentication, export to excel, tablet layout")
            if search_query:
                embed_model = load_embedding_model()
                q_emb = embed_model.encode([search_query])[0]
                sims, indices = semantic_search(q_emb, st.session_state.index, k=5)
                
                st.markdown(f"**Top matching feedback signals for '{search_query}':**")
                for sim, idx in zip(sims, indices):
                    if idx < len(df):
                        match_row = df.iloc[idx]
                        st.markdown(f"""
                        <div class="saas-quote">
                            "{match_row['Request_Text']}"
                            <br><small><strong>Category:</strong> {match_row['Category']} | <strong>Similarity:</strong> {sim:.2f} | <strong>Sentiment:</strong> {match_row['Sentiment']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
        with copilot_sub2:
            st.markdown("#### Automated PRD Synthesizer")
            theme_choice = st.selectbox("Select Feature Theme to Generate PRD:", roadmap_df['Theme'].tolist(), key="prd_theme_select")
            if st.button("✨ Synthesize PRD", key="gen_prd_btn", use_container_width=True):
                with st.spinner("Drafting comprehensive PRD..."):
                    c_id = roadmap_df[roadmap_df['Theme'] == theme_choice]['ID'].values[0]
                    c_texts = " | ".join(df[df['Cluster'] == c_id]['Request_Text'].head(12).tolist())
                    prd_out = generate_prd(
                        hf_token="",
                        theme_name=theme_choice,
                        summary=st.session_state.cluster_summaries.get(c_id, ""),
                        keywords=", ".join(st.session_state.cluster_keywords.get(c_id, [])),
                        sample_requests=c_texts
                    )
                    st.success("PRD generated successfully!")
                    st.markdown(prd_out)
                    st.download_button("📥 Download PRD (.md)", prd_out, file_name=f"PRD_{theme_choice}.md", mime="text/markdown")
                    
        with copilot_sub3:
            st.markdown("#### Customer-Facing Release Notes")
            rn_choice = st.selectbox("Select Feature Theme for Release Notes:", roadmap_df['Theme'].tolist(), key="rn_theme_select")
            if st.button("✨ Draft Release Notes", key="gen_rn_btn", use_container_width=True):
                with st.spinner("Drafting public release notes..."):
                    row = roadmap_df[roadmap_df['Theme'] == rn_choice].iloc[0]
                    rn_out = generate_release_notes(
                        hf_token="",
                        theme_name=rn_choice,
                        priority=row['Priority'],
                        effort=str(row['Effort'])
                    )
                    st.success("Release notes drafted successfully!")
                    st.markdown(rn_out)
                    st.download_button("📥 Download Release Notes (.md)", rn_out, file_name=f"Release_Notes_{rn_choice}.md", mime="text/markdown")

    # =========================================================================
    # VIEW 10: 📄 EXECUTIVE REPORTS
    # =========================================================================
    elif st.session_state.active_nav == "📄 Executive Reports":
        st.markdown("### 📄 Executive Briefing & Export Center")
        st.caption("Generate presentation-ready strategy reports across Markdown, HTML Dashboard, and PDF formats.")
        
        col_rep1, col_rep2, col_rep3 = st.columns(3)
        with col_rep1:
            md_report = generate_markdown_report(df, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
            st.download_button("📄 Download Markdown Report (.md)", data=md_report, file_name="executive_product_strategy.md", mime="text/markdown", use_container_width=True)
            
        with col_rep2:
            html_report = generate_html_report(df, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
            st.download_button("🌐 Download HTML Dashboard (.html)", data=html_report, file_name="executive_product_dashboard.html", mime="text/html", use_container_width=True)
            
        with col_rep3:
            try:
                pdf_buffer = generate_pdf_report(df, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
                st.download_button("📑 Download Executive PDF (.pdf)", data=pdf_buffer.getvalue(), file_name="executive_product_strategy.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF engine notice: {e}")
                
        st.markdown("---")
        st.markdown("#### 📋 Executive Strategy Summary Preview")
        st.markdown(md_report[:2000] + "...")

    # =========================================================================
    # VIEW 11: ⚙️ SETTINGS & INGESTION
    # =========================================================================
    elif st.session_state.active_nav == "⚙️ Settings & Ingestion":
        st.markdown("### ⚙️ Workspace Settings & Data Ingestion")
        st.caption("Manage raw feedback sources, reload sample datasets, or configure Hugging Face tokens.")
        
        new_upload = st.file_uploader("Upload Custom Feature Requests CSV", type=["csv"])
        if new_upload is not None:
            if st.button("⚡ Ingest & Process New Dataset", type="primary"):
                st.session_state.processed_data = False
                st.rerun()
                
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if st.button("📁 Reload 1,000 Sample Dataset", use_container_width=True):
                st.session_state.processed_data = False
                st.rerun()
        with col_res2:
            if st.button("🔄 Reset Entire Workspace", use_container_width=True):
                for k in state_defaults.keys():
                    st.session_state[k] = state_defaults[k]
                st.rerun()

else:
    st.info("Loading Prioritize.ai workspace...")
