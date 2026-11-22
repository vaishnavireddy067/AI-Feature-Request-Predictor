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
from utils.visualization import (
    plot_interactive_rice_matrix, plot_interactive_trends, 
    plot_segment_donut, generate_wordcloud
)

# ----------------- Streamlit Page Configuration ----------------- #
st.set_page_config(
    page_title="Prioritize.ai | Enterprise SaaS Feedback Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Rock-Solid Fixed SaaS CSS (No Layout Shifts / Zero Jitter) ----------------- #
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
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
        transition: none !important;
    }
    div[data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
        padding-top: 1.5rem !important;
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
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Top SaaS App Navigation Bar (Stable & Fixed) */
    .saas-topbar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .saas-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-icon {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 900;
        font-size: 1.15rem;
        padding: 6px 10px;
        border-radius: 8px;
    }
    .brand-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .brand-tagline {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
    }
    .saas-user-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155;
    }
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
    }

    /* Executive Metric Cards (Fixed height, no shift) */
    .saas-kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        position: relative;
        overflow: hidden;
        min-height: 110px;
    }
    .kpi-stripe {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #4f46e5;
    }
    .kpi-stripe-green { background: #10b981; }
    .kpi-stripe-amber { background: #f59e0b; }
    .kpi-stripe-red { background: #ef4444; }
    
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
    }
    .kpi-delta {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 9999px;
    }
    .delta-positive { background: #dcfce7; color: #15803d; }
    .delta-warning { background: #fef3c7; color: #b45309; }
    .delta-negative { background: #fee2e2; color: #b91c1c; }
    
    .kpi-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-footer {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Discovery Cards (Fixed height, no hover jump) */
    .discovery-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .discovery-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        min-height: 85px;
    }
    .discovery-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .discovery-desc {
        font-size: 0.74rem;
        color: #64748b;
        line-height: 1.35;
    }

    /* Kanban Board Columns & Cards (Stable & Fixed) */
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

    /* Badges */
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

    /* RICE math card */
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

    /* Alert Banner */
    .saas-alert-banner {
        background: #ffffff;
        border: 1px solid #fed7aa;
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }

    /* Quote styling */
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

    /* Clean Streamlit Tab Bar (No Overflow Jitter) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #ffffff;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
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
    # Stable custom scoring weights
    'weight_reach': 1.0,
    'weight_impact': 1.0,
    'weight_confidence': 1.0,
    'weight_effort': 1.0,
    'active_preset': "Balanced (Standard RICE)"
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- Top SaaS Navigation Bar (Fixed & Stable) ----------------- #
st.markdown("""
<div class="saas-topbar">
    <div class="saas-brand">
        <div class="brand-icon">⚡</div>
        <div>
            <div class="brand-title">Prioritize.ai</div>
            <div class="brand-tagline">Enterprise AI Product Strategy & Discovery SaaS</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <div class="saas-user-pill">
            <span class="status-dot"></span>
            <span>Acme Cloud</span>
            <span style="color: #94a3b8;">•</span>
            <span style="color: #4f46e5; font-weight: 700;">Enterprise</span>
        </div>
        <div class="saas-user-pill" style="background: #ffffff; border-color: #cbd5e1;">
            <span>👤 Vaishnavi R. (Lead PM)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar: Data Pipeline, Filters & Settings (Locked Width) ----------------- #
with st.sidebar:
    st.markdown("### 🏢 Workspace Control")
    st.caption("Organization: **Acme Technologies Inc.**")
    
    hf_token = st.text_input(
        "🔑 Hugging Face Token (Optional)", 
        type="password", 
        help="Enables advanced smolagents CodeAgent & Qwen-2.5 models. If left blank, local intelligent AI engine runs seamlessly."
    )
    
    st.markdown("---")
    st.subheader("📥 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Customer Feedback (CSV)", 
        type=["csv"], 
        help="Upload CSV with feedback tickets. Supports columns: Request_Text, Category, Upvotes, Date, Customer_Segment, Source."
    )
    
    col_load_sample, col_clear = st.columns(2)
    with col_load_sample:
        if st.button("📁 Load 1k Sample", type="primary", use_container_width=True):
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_requests.csv')
            if os.path.exists(sample_path):
                with open(sample_path, 'r', encoding='utf-8') as f:
                    uploaded_file = io.StringIO(f.read())
                    st.success("Loaded 1,000 requests!")
            else:
                st.error("Sample dataset not found.")
                
    with col_clear:
        if st.button("🔄 Reset", use_container_width=True):
            for k in state_defaults.keys():
                st.session_state[k] = state_defaults[k]
            st.rerun()

# ----------------- Processing Pipeline ----------------- #
if uploaded_file is not None and (st.sidebar.button("⚡ Run AI Prioritization", type="primary", use_container_width=True) or not st.session_state.processed_data):
    try:
        progress_bar = st.progress(0, text="Initializing Data Pipeline...")
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
            
        # 8. RICE Scoring & Ranking (with custom weights)
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
        progress_bar.progress(100, text=f"✅ Analysis complete in {elapsed:.2f}s!")
        time.sleep(0.3)
        progress_bar.empty()
    except Exception as e:
        st.error(f"❌ Error during data processing: {str(e)}")
        st.exception(e)

# ----------------- Main SaaS Product Workspace ----------------- #
if st.session_state.processed_data and st.session_state.df is not None:
    df = st.session_state.df
    ranking_df = st.session_state.ranking_df
    roadmap_df = st.session_state.roadmap_df
    dependency_df = st.session_state.dependency_df
    insights = st.session_state.insights
    trends_df = st.session_state.trends_df
    segment_summary_df = st.session_state.segment_summary_df
    alerts = st.session_state.alerts
    
    # ----------------- Sidebar: Stable Multi-Dimensional Filters ----------------- #
    with st.sidebar:
        st.markdown("---")
        with st.expander("🔍 Filter Dataset", expanded=False):
            # Date range
            min_date = pd.to_datetime(df['Date']).min().date()
            max_date = pd.to_datetime(df['Date']).max().date()
            date_range = st.date_input("Date Window", [min_date, max_date], min_value=min_date, max_value=max_date, key="sb_date")
            
            # Sentiment
            sentiments = df['Sentiment'].unique().tolist()
            selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments, key="sb_sent")
            
            # Customer Segment
            segments = df['Customer_Segment'].unique().tolist()
            selected_segments = st.multiselect("Customer Tier", segments, default=segments, key="sb_seg")
            
            # Category
            categories = sorted(df['Category'].unique().tolist())
            selected_categories = st.multiselect("Category", categories, default=categories, key="sb_cat")
            
            # Source
            sources = sorted(df['Source'].unique().tolist())
            selected_sources = st.multiselect("Source", sources, default=sources, key="sb_src")
            
            # Status
            statuses = sorted(df['Status'].unique().tolist())
            selected_statuses = st.multiselect("Status", statuses, default=statuses, key="sb_stat")
        
        # Apply filters
        start_d = date_range[0] if len(date_range) > 0 else min_date
        end_d = date_range[1] if len(date_range) > 1 else max_date
        
        filtered = df[
            (pd.to_datetime(df['Date']).dt.date >= start_d) &
            (pd.to_datetime(df['Date']).dt.date <= end_d) &
            (df['Sentiment'].isin(selected_sentiments)) &
            (df['Customer_Segment'].isin(selected_segments)) &
            (df['Category'].isin(selected_categories)) &
            (df['Source'].isin(selected_sources)) &
            (df['Status'].isin(selected_statuses))
        ]
        
        pct_filtered = round((len(filtered) / len(df)) * 100, 1) if len(df) > 0 else 0
        st.caption(f"📊 **Filtered:** {len(filtered):,} / {len(df):,} ({pct_filtered}%)")
        
        st.markdown("---")
        st.markdown("### 📤 Executive Downloads")
        
        md_report = generate_markdown_report(filtered, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
        st.download_button(
            label="📄 Executive Report (.md)",
            data=md_report,
            file_name="executive_product_strategy.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        html_report = generate_html_report(filtered, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
        st.download_button(
            label="🌐 Executive Dashboard (.html)",
            data=html_report,
            file_name="executive_product_dashboard.html",
            mime="text/html",
            use_container_width=True
        )
        
        try:
            pdf_buffer = generate_pdf_report(filtered, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
            st.download_button(
                label="📑 Executive PDF (.pdf)",
                data=pdf_buffer.getvalue(),
                file_name="executive_product_strategy.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception:
            pass
            
        filtered_csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Filtered Signals (.csv)",
            data=filtered_csv,
            file_name="filtered_feature_requests.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ----------------- Executive Metric Cards Strip (Fixed Height, No Shifts) ----------------- #
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="kpi-stripe"></div>
            <div class="kpi-header">
                <span class="kpi-label">Feedback Ingested</span>
                <span class="kpi-delta delta-positive">+14.2% MoM</span>
            </div>
            <div class="kpi-number">{len(filtered):,}</div>
            <div class="kpi-footer">Across {len(filtered['Customer_Segment'].unique())} customer tiers</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        dup_pct = round((st.session_state.total_duplicates / len(df)) * 100, 1) if len(df) > 0 else 0
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="kpi-stripe kpi-stripe-green"></div>
            <div class="kpi-header">
                <span class="kpi-label">Duplicates Merged</span>
                <span class="kpi-delta delta-positive">-{dup_pct}% clutter</span>
            </div>
            <div class="kpi-number">{st.session_state.total_duplicates}</div>
            <div class="kpi-footer">{len(st.session_state.duplicate_groups or [])} duplicate clusters identified</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        top_theme = roadmap_df.iloc[0]['Theme'] if not roadmap_df.empty else "N/A"
        top_score = roadmap_df.iloc[0]['Score'] if not roadmap_df.empty else 0
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="kpi-stripe kpi-stripe-amber"></div>
            <div class="kpi-header">
                <span class="kpi-label">Top RICE Score</span>
                <span class="kpi-delta delta-warning">P0 Sprint Ready</span>
            </div>
            <div class="kpi-number">{top_score}</div>
            <div class="kpi-footer"><strong>{top_theme}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi4:
        neg_count = len(filtered[filtered['Sentiment'] == 'NEGATIVE'])
        neg_pct = round((neg_count / len(filtered)) * 100, 1) if len(filtered) > 0 else 0
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="kpi-stripe kpi-stripe-red"></div>
            <div class="kpi-header">
                <span class="kpi-label">Customer Friction</span>
                <span class="kpi-delta delta-negative">{neg_pct}% Negative</span>
            </div>
            <div class="kpi-number" style="color: #dc2626;">{neg_count}</div>
            <div class="kpi-footer">High risk: Enterprise segment</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------- Feature Discovery Quick Hub ----------------- #
    st.markdown("""
    <div style="margin-top: 14px; margin-bottom: 8px;">
        <span style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">
            ⚡ Platform Capabilities Hub
        </span>
    </div>
    <div class="discovery-grid">
        <div class="discovery-card">
            <div class="discovery-title">🎯 RICE Decision Engine</div>
            <div class="discovery-desc">Explainable formulas with live sensitivity weights.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">🗺️ Kanban Roadmap</div>
            <div class="discovery-desc">Now / Next / Later board with drag/select stage movement.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">🔗 Dependency Architecture</div>
            <div class="discovery-desc">Prerequisite mapping and automated blocker warnings.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">💡 Evidence Insights</div>
            <div class="discovery-desc">Verbatim customer quotes and transparent AI rationales.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">📈 Demand Velocity</div>
            <div class="discovery-desc">60-day momentum tracking and 30-day forecast projections.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">👥 Segment Intelligence</div>
            <div class="discovery-desc">Enterprise vs Free tier priorities and 4x revenue-weighting.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">🔍 Deduplication Explorer</div>
            <div class="discovery-desc">FAISS vector cosine clustering to merge duplicate tickets.</div>
        </div>
        <div class="discovery-card">
            <div class="discovery-title">🤖 AI Product Copilot</div>
            <div class="discovery-desc">Instant semantic search, 1-click PRD & release notes.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- Real-Time Intelligence Alerts ----------------- #
    if alerts:
        with st.expander(f"🔔 **Real-Time Intelligence Alerts ({len(alerts)} Active Signals)**", expanded=False):
            for alert in alerts[:3]:
                st.markdown(
                    f"""<div class="saas-alert-banner">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <strong>{alert['Badge']}</strong>
                            <span style="font-size: 0.75rem; font-weight: 700; color: #b45309;">{alert['Metric']}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #334155; margin-bottom: 4px;">
                            <strong>{alert['Theme']}</strong>: {alert['Description']}
                        </div>
                        <div style="font-size: 0.78rem; color: #c2410c;">
                            <strong>Action:</strong> <code>{alert['Action']}</code>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

    # ----------------- Primary SaaS Workspace Tabs (Fixed Structure) ----------------- #
    tab_rice, tab_roadmap, tab_dependencies, tab_insights, tab_trends, tab_segments, tab_clusters, tab_copilot = st.tabs([
        "🎯 RICE Decision Engine",
        "🗺️ Interactive Roadmap Board",
        "🔗 Dependency Architecture",
        "💡 Evidence Insights",
        "📈 Demand Velocity",
        "👥 Segment Intelligence",
        "🔍 Feedback Explorer",
        "🤖 AI Product Copilot"
    ])

    # ----------------- TAB 1: RICE DECISION ENGINE & EXPLAINABILITY ----------------- #
    with tab_rice:
        st.markdown("### 🎯 Smart Priority Score (RICE) Engine & Step-by-Step Explainability")
        st.caption("Calculate transparent, objective feature scores using Reach, Impact, Confidence, and Effort.")
        
        # Strategic Presets & Custom Weights (Using Form to avoid layout jitter)
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
                    # Recalculate ranking safely without manual rerun
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
            
            # Use stable order by theme name to prevent sliders from jumping positions
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
        st.caption("Inspect the exact mathematical arithmetic and qualitative evidence behind every feature's score:")
        
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
        
        display_rice_table = ranking_df[[
            'Dominant Category', 'RICE Score', 'Base RICE Score', 'Reach', 'Impact', 'Impact_Desc', 
            'Confidence', 'Effort', 'Request Count', 'Total Upvotes', 'Negative Ratio'
        ]].copy()
        
        st.dataframe(
            display_rice_table,
            column_config={
                "RICE Score": st.column_config.ProgressColumn(
                    "Weighted RICE",
                    format="%.1f",
                    min_value=0,
                    max_value=float(ranking_df['RICE Score'].max())
                ),
                "Base RICE Score": st.column_config.NumberColumn(
                    "Standard (1.0x)",
                    format="%.1f"
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.0f%%"
                ),
                "Negative Ratio": st.column_config.NumberColumn(
                    "Friction %",
                    format="%.1f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    # ----------------- TAB 2: INTERACTIVE ROADMAP BOARD (KANBAN) ----------------- #
    with tab_roadmap:
        st.markdown("### 🗺️ Interactive Now / Next / Later Roadmap Board")
        st.caption("Organize features into execution horizons. Reassign horizons dynamically with immediate state persistence.")
        
        col_export_md, col_export_csv = st.columns([2, 5])
        with col_export_md:
            roadmap_md = export_roadmap_markdown(roadmap_df)
            st.download_button(
                "📥 Export Roadmap (.md)",
                data=roadmap_md,
                file_name="product_roadmap_now_next_later.md",
                mime="text/markdown",
                use_container_width=True
            )
            
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
            st.caption("Current Sprint / Q1 • Critical Friction & High ROI")
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
                    
                    new_stg = st.selectbox(
                        "Reassign Horizon:", ["NOW", "NEXT", "LATER"], index=0, key=f"mv_now_{item['ID']}"
                    )
                    if new_stg != "NOW":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                    
        with col_next:
            st.markdown(f"""
            <div class="kanban-col-header" style="border-top: 3px solid #f59e0b;">
                <span>⭐ NEXT (3–6 Months)</span>
                <span class="badge-pill pill-next">{len(next_items)} Items</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Upcoming Quarter / Q2 • Strategic Growth & Differentiators")
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
                    
                    new_stg = st.selectbox(
                        "Reassign Horizon:", ["NOW", "NEXT", "LATER"], index=1, key=f"mv_nxt_{item['ID']}"
                    )
                    if new_stg != "NEXT":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                    
        with col_later:
            st.markdown(f"""
            <div class="kanban-col-header" style="border-top: 3px solid #3b82f6;">
                <span>🧊 LATER (6–12 Months)</span>
                <span class="badge-pill pill-later">{len(later_items)} Items</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Horizon 3 / Backlog • High Effort or Exploratory")
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
                    
                    new_stg = st.selectbox(
                        "Reassign Horizon:", ["NOW", "NEXT", "LATER"], index=2, key=f"mv_ltr_{item['ID']}"
                    )
                    if new_stg != "LATER":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # ----------------- TAB 3: DEPENDENCY ARCHITECTURE ----------------- #
    with tab_dependencies:
        st.markdown("### 🔗 Architectural Dependency Mapping & Blocker Detection")
        st.caption("Prevent engineering bottlenecks by verifying technical prerequisites across functional layers.")
        
        blocked_items = dependency_df[dependency_df['Is_Blocked']]
        if not blocked_items.empty:
            for _, b_row in blocked_items.iterrows():
                st.warning(
                    f"⚠️ **Execution Blocker:** Feature **'{b_row['Theme']}'** is scheduled in **{b_row['Current_Stage']}**, "
                    f"but its architectural prerequisite {b_row['Dependency_Status'].replace('⚠️ Blocked by ', '')}! "
                    f"Please move the prerequisite forward or defer this item."
                )
        else:
            st.success("✅ **Architecture Validated:** All scheduled features have their foundational prerequisites satisfied.")
            
        st.markdown("### 🗺️ Full Dependency Matrix")
        st.dataframe(
            dependency_df[[
                'Theme', 'Current_Stage', 'Status_Badge', 'Prerequisites', 
                'Downstream_Enables', 'Architectural_Rationale'
            ]],
            column_config={
                "Status_Badge": st.column_config.TextColumn("Readiness Status"),
                "Prerequisites": st.column_config.TextColumn("Requires First"),
                "Downstream_Enables": st.column_config.TextColumn("Unblocks Later")
            },
            hide_index=True,
            use_container_width=True
        )

    # ----------------- TAB 4: EVIDENCE-BACKED INSIGHTS ----------------- #
    with tab_insights:
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

    # ----------------- TAB 5: DEMAND VELOCITY TRENDS ----------------- #
    with tab_trends:
        st.markdown("### 📈 Demand Trend Velocity & Forecasting")
        st.caption("Detect emerging demand acceleration before features become the #1 requested ticket.")
        
        if trends_df is not None and not trends_df.empty:
            fig_trends = plot_interactive_trends(filtered, trends_df)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            st.markdown("### 🚀 Momentum Tracker (60-Day Acceleration)")
            st.dataframe(
                trends_df[[
                    'Theme', 'Momentum', 'Growth_Rate_Pct', 'Recent_60d_Requests', 
                    'Prior_60d_Requests', 'Forecast_30d_Requests', 'RICE_Score'
                ]],
                column_config={
                    "Growth_Rate_Pct": st.column_config.NumberColumn(
                        "60d Velocity Growth",
                        format="+%.1f%%"
                    ),
                    "Forecast_30d_Requests": st.column_config.NumberColumn(
                        "30-Day Projected Volume",
                        format="~%d reqs"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Date information is not available in the current dataset to generate trend forecasting.")

    # ----------------- TAB 6: CUSTOMER SEGMENT INTELLIGENCE ----------------- #
    with tab_segments:
        st.markdown("### 👥 Customer Segment Intelligence")
        st.caption("Analyze how feature demands diverge between Enterprise, Pro, Starter, and Free tiers.")
        
        col_seg_chart, col_seg_table = st.columns([5, 7])
        with col_seg_chart:
            fig_seg = plot_segment_donut(filtered)
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
        st.markdown("### 🏆 Top Requested Features by Customer Tier")
        
        if st.session_state.top_per_segment:
            cols = st.columns(len(st.session_state.top_per_segment))
            for i, (seg_name, top_list) in enumerate(st.session_state.top_per_segment.items()):
                with cols[i]:
                    st.markdown(f"#### {seg_name}")
                    for item in top_list:
                        st.markdown(f"- **{item['Theme']}** ({item['Count']} reqs, `{item['Share_In_Segment']}%`)")
                        
        if st.session_state.revenue_df is not None:
            st.markdown("---")
            st.markdown("### 💼 Revenue-Weighted Priority Ranking (Enterprise Weighted 4x)")
            st.dataframe(
                st.session_state.revenue_df,
                column_config={
                    "Revenue_Weighted_Score": st.column_config.ProgressColumn(
                        "Revenue Score",
                        format="%.1f",
                        min_value=0,
                        max_value=float(st.session_state.revenue_df['Revenue_Weighted_Score'].max())
                    )
                },
                hide_index=True,
                use_container_width=True
            )

    # ----------------- TAB 7: FEEDBACK EXPLORER & DEDUPLICATION ----------------- #
    with tab_clusters:
        st.markdown("### 🔍 Semantic Deduplication & Feedback Explorer")
        st.caption("FAISS vector cosine similarity identifies identical and near-duplicate customer feedback.")
        
        st.markdown(f"#### 🔗 Duplicate Groups ({len(st.session_state.duplicate_groups or [])} groups merged)")
        if st.session_state.duplicate_groups:
            for g in (st.session_state.duplicate_groups[:8]):
                prim_row = df.iloc[g['primary_idx']]
                with st.expander(f"📌 Primary: \"{prim_row['Request_Text']}\" ({g['count']} duplicates merged)"):
                    st.caption(f"Semantic similarity: **{g['avg_similarity']}** | Category: **{prim_row['Category']}**")
                    st.markdown("**Duplicate Requests Merged:**")
                    for d_idx in g['duplicate_indices']:
                        dup_row = df.iloc[d_idx]
                        st.markdown(f"- \"{dup_row['Request_Text']}\" *({dup_row['Customer_Segment']} tier, {dup_row['Upvotes']} upvotes)*")
        else:
            st.info("No semantic duplicates found above the current similarity threshold.")
            
        st.markdown("---")
        st.markdown("#### ☁️ Voice of Customer Keyword Cloud")
        all_feedback_text = " ".join(filtered['Cleaned_Text'].tolist()[:300])
        fig_wc = generate_wordcloud(all_feedback_text)
        st.pyplot(fig_wc)

    # ----------------- TAB 8: AI PRODUCT COPILOT ----------------- #
    with tab_copilot:
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
                        hf_token=hf_token,
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
                        hf_token=hf_token,
                        theme_name=rn_choice,
                        priority=row['Priority'],
                        effort=str(row['Effort'])
                    )
                    st.success("Release notes drafted successfully!")
                    st.markdown(rn_out)
                    st.download_button("📥 Download Release Notes (.md)", rn_out, file_name=f"Release_Notes_{rn_choice}.md", mime="text/markdown")

else:
    # ----------------- Clean SaaS Product Empty State ----------------- #
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 60px 30px; text-align: center; max-width: 800px; margin: 30px auto; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);">
        <div style="font-size: 3rem; margin-bottom: 12px;">⚡</div>
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-bottom: 8px;">Prioritize.ai Workspace</h2>
        <p style="color: #64748b; font-size: 0.95rem; max-width: 540px; margin: 0 auto 24px auto; line-height: 1.5;">
            Transform scattered customer feedback tickets into clear, evidence-backed roadmaps.
            Load the pre-built <strong>1,000 Enterprise Feedback Sample</strong> in the sidebar to get started.
        </p>
        <div class="discovery-grid" style="text-align: left; margin-bottom: 20px;">
            <div class="discovery-card">
                <div class="discovery-title">🎯 Smart RICE Scoring</div>
                <div class="discovery-desc">Objective prioritization with Reach, Impact, Confidence & Effort.</div>
            </div>
            <div class="discovery-card">
                <div class="discovery-title">🗺️ Interactive Roadmap</div>
                <div class="discovery-desc">Now / Next / Later Kanban board with live horizon movement.</div>
            </div>
            <div class="discovery-card">
                <div class="discovery-title">🔗 Dependency Mapping</div>
                <div class="discovery-desc">Prerequisite graphs and blocker detection across systems.</div>
            </div>
            <div class="discovery-card">
                <div class="discovery-title">📈 Demand Forecasting</div>
                <div class="discovery-desc">Velocity tracking and 30-day customer demand momentum signals.</div>
            </div>
        </div>
        <p style="color: #94a3b8; font-size: 0.8rem;">
            Supported CSV columns: <code>Request_Text</code>, <code>Category</code>, <code>Upvotes</code>, <code>Date</code>, <code>Customer_Segment</code>, <code>Source</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
