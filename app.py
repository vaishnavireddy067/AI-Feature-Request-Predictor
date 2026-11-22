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
    plot_segment_donut, plot_cluster_distribution, generate_wordcloud
)

# ----------------- Streamlit Page Configuration ----------------- #
st.set_page_config(
    page_title="AI Feature Request Prioritizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom Styling (Clean Modern SaaS) ----------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Global Card Styles */
    .saas-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    
    /* KPI Metric Boxes */
    .kpi-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-left: 4px solid #4f46e5;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Badges */
    .badge-now {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-next {
        background-color: #fef3c7;
        color: #d97706;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-later {
        background-color: #e0f2fe;
        color: #0284c7;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-alert {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    /* Insight Quote Box */
    .quote-card {
        background-color: #f8fafc;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-style: italic;
        font-size: 0.88rem;
        color: #334155;
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
    # Custom scoring weights
    'weight_reach': 1.0,
    'weight_impact': 1.0,
    'weight_confidence': 1.0,
    'weight_effort': 1.0,
    'active_preset': "Balanced (Standard RICE)"
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- App Header ----------------- #
col_head, col_badge = st.columns([4, 1])
with col_head:
    st.title("🚀 AI Feature Request Prioritizer")
    st.markdown("Automate customer feedback analysis, detect duplicates, calculate explainable **RICE priority scores**, map dependencies, and build dynamic Now / Next / Later roadmaps.")
with col_badge:
    st.markdown("<div style='text-align: right; padding-top: 15px;'><span class='badge-now'>Sprint 1-3 Active</span></div>", unsafe_allow_html=True)

# ----------------- Sidebar: Data Pipeline & Filters ----------------- #
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    hf_token = st.text_input("🔑 Hugging Face Token (Optional)", type="password", help="Enables advanced smolagents CodeAgent & Qwen-2.5 models. If blank, local intelligent AI engine runs seamlessly.")
    
    uploaded_file = st.file_uploader("Upload Feature Requests (CSV)", type=["csv"], help="Upload customer feedback CSV. Supports columns like Request_Text, Category, Upvotes, Date, Customer_Segment, Source.")
    
    col_load_sample, col_clear = st.columns(2)
    with col_load_sample:
        if st.button("📁 Sample Data", help="Load pre-built 1,000 enterprise feedback dataset."):
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_requests.csv')
            if os.path.exists(sample_path):
                with open(sample_path, 'r', encoding='utf-8') as f:
                    uploaded_file = io.StringIO(f.read())
                    st.success("Sample data loaded!")
            else:
                st.error("Sample dataset not found.")
                
    with col_clear:
        if st.button("🔄 Reset App"):
            for k in state_defaults.keys():
                st.session_state[k] = state_defaults[k]
            st.rerun()

# ----------------- Processing Pipeline ----------------- #
if uploaded_file is not None and (st.sidebar.button("⚡ Run AI Analysis", type="primary") or not st.session_state.processed_data):
    try:
        progress_bar = st.progress(0, text="Initializing Data Pipeline...")
        start_time = time.time()
        
        # 1. Clean & Standardize
        progress_bar.progress(10, text="Standardizing and cleaning feedback text...")
        df = load_and_clean_data(uploaded_file)
        
        # 2. Sentiment Analysis
        progress_bar.progress(25, text="Analyzing customer sentiment and friction points...")
        sentiment_model = load_sentiment_model()
        df['Sentiment'] = analyze_sentiments(df['Cleaned_Text'].tolist(), sentiment_model)
        
        # 3. Auto-Tag Cross-functional Teams
        progress_bar.progress(35, text="Tagging functional domains (Backend, Frontend, Mobile, Security)...")
        df = tag_dataframe(df)
        
        # 4. Vector Embeddings
        progress_bar.progress(50, text="Computing dense semantic embeddings (all-MiniLM-L6-v2)...")
        embed_model = load_embedding_model()
        embeddings = generate_embeddings(df['Cleaned_Text'].tolist(), embed_model)
        
        # 5. Semantic Deduplication & Indexing
        progress_bar.progress(65, text="Building cosine similarity index & detecting duplicates...")
        index = build_faiss_index(embeddings)
        dup_groups = find_duplicates(index, embeddings, similarity_threshold=0.82)
        total_dups = sum(g['count'] - 1 for g in dup_groups)
        
        # 6. KMeans Semantic Clustering
        progress_bar.progress(75, text="Clustering requests into strategic themes...")
        n_clusters = min(12, max(2, len(df) // 25))
        df['Cluster'] = perform_clustering(embeddings, n_clusters=n_clusters)
        
        # 7. Keywords & Extractive Centroid Summaries
        progress_bar.progress(85, text="Extracting keywords and representative feedback...")
        kw_model = load_keyword_model()
        cluster_summaries = {}
        cluster_keywords = {}
        for c_id in sorted(df['Cluster'].unique()):
            c_texts = df[df['Cluster'] == c_id]['Cleaned_Text'].tolist()
            cluster_summaries[c_id] = summarize_cluster(c_texts, embed_model)
            cluster_keywords[c_id] = extract_keywords(" ".join(c_texts[:50]), kw_model, top_n=5)
            
        # 8. RICE Scoring & Ranking (using custom weights)
        progress_bar.progress(90, text="Calculating Smart RICE Priority Scores...")
        ranking_df = rank_features(
            df, cluster_col='Cluster', upvotes_col='Upvotes',
            weight_reach=st.session_state.weight_reach,
            weight_impact=st.session_state.weight_impact,
            weight_confidence=st.session_state.weight_confidence,
            weight_effort=st.session_state.weight_effort
        )
        
        # 9. Strategic Artifacts: Roadmap, Insights, Dependencies, Trends, Segments, Alerts
        progress_bar.progress(95, text="Generating Roadmap, Dependencies, and Emerging Alerts...")
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
        progress_bar.progress(100, text=f"✅ Analysis Complete in {elapsed:.2f}s!")
        time.sleep(0.4)
        progress_bar.empty()
    except Exception as e:
        st.error(f"❌ Error during data processing: {str(e)}")
        st.exception(e)

# ----------------- Main Dashboard Interface ----------------- #
if st.session_state.processed_data and st.session_state.df is not None:
    df = st.session_state.df
    ranking_df = st.session_state.ranking_df
    roadmap_df = st.session_state.roadmap_df
    dependency_df = st.session_state.dependency_df
    insights = st.session_state.insights
    trends_df = st.session_state.trends_df
    segment_summary_df = st.session_state.segment_summary_df
    alerts = st.session_state.alerts
    
    # ----------------- Advanced Multi-Criteria Filters in Sidebar ----------------- #
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Advanced Filters")
        
        # 1. Date Range Filter
        min_date = pd.to_datetime(df['Date']).min().date()
        max_date = pd.to_datetime(df['Date']).max().date()
        date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        # 2. Sentiment Filter
        sentiments = df['Sentiment'].unique().tolist()
        selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
        
        # 3. Customer Segment Filter
        segments = df['Customer_Segment'].unique().tolist()
        selected_segments = st.multiselect("Customer Segment", segments, default=segments)
        
        # 4. Category Filter
        categories = sorted(df['Category'].unique().tolist())
        selected_categories = st.multiselect("Category", categories, default=categories)
        
        # 5. Source Channel Filter
        sources = sorted(df['Source'].unique().tolist())
        selected_sources = st.multiselect("Feedback Source", sources, default=sources)
        
        # 6. Status Filter
        statuses = sorted(df['Status'].unique().tolist())
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
        
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
        
        pct_filtered = round((len(filtered) / len(df)) * 100, 1)
        st.caption(f"📊 **Showing {len(filtered):,} of {len(df):,} requests ({pct_filtered}%)**")
        
        # ----------------- Executive Report Downloads in Sidebar ----------------- #
        st.markdown("---")
        st.subheader("📥 Executive Downloads")
        
        # Markdown report
        md_report = generate_markdown_report(filtered, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
        st.download_button(
            label="📄 Executive Report (.md)",
            data=md_report,
            file_name="executive_product_strategy.md",
            mime="text/markdown"
        )
        
        # HTML report
        html_report = generate_html_report(filtered, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
        st.download_button(
            label="🌐 Executive Dashboard (.html)",
            data=html_report,
            file_name="executive_product_dashboard.html",
            mime="text/html"
        )
        
        # PDF report via ReportLab
        try:
            pdf_buffer = generate_pdf_report(filtered, ranking_df, roadmap_df, insights, trends_df, segment_summary_df)
            st.download_button(
                label="📑 Executive PDF Report (.pdf)",
                data=pdf_buffer.getvalue(),
                file_name="executive_product_strategy.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.caption(f"PDF export notice: {e}")
            
        # CSV Export
        filtered_csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Export Filtered Requests (.csv)",
            data=filtered_csv,
            file_name="filtered_feature_requests.csv",
            mime="text/csv"
        )

    # ----------------- Emerging Feature Alerts Banner ----------------- #
    if alerts:
        with st.expander(f"🚨 **Emerging Feature Alerts ({len(alerts)} Signals Detected)**", expanded=True):
            for alert in alerts[:3]:
                st.markdown(
                    f"""<div class="badge-alert">
                        <strong>{alert['Badge']}</strong>: <strong>{alert['Theme']}</strong> — {alert['Metric']}
                        <br><span style="color: #475569;">{alert['Description']}</span>
                        <br><strong>Recommended Action:</strong> <code>{alert['Action']}</code>
                    </div>""",
                    unsafe_allow_html=True
                )

    # ----------------- Top Level KPI Banner ----------------- #
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total Feedback</div>
            <div class="kpi-value">{len(filtered):,}</div>
            <div class="kpi-sub">Across {len(filtered['Customer_Segment'].unique())} user segments</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #10b981;">
            <div class="kpi-title">Duplicates Merged</div>
            <div class="kpi-value">{st.session_state.total_duplicates}</div>
            <div class="kpi-sub">{len(st.session_state.duplicate_groups or [])} duplicate clusters identified</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        top_theme = roadmap_df.iloc[0]['Theme'] if not roadmap_df.empty else "N/A"
        top_score = roadmap_df.iloc[0]['Score'] if not roadmap_df.empty else 0
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #f59e0b;">
            <div class="kpi-title">Top RICE Priority</div>
            <div class="kpi-value">{top_score}</div>
            <div class="kpi-sub">{top_theme}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        neg_count = len(filtered[filtered['Sentiment'] == 'NEGATIVE'])
        neg_pct = round((neg_count / len(filtered)) * 100, 1) if len(filtered) > 0 else 0
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #ef4444;">
            <div class="kpi-title">Friction Points</div>
            <div class="kpi-value" style="color: #dc2626;">{neg_count}</div>
            <div class="kpi-sub">{neg_pct}% negative customer sentiment</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------- Primary Application Tabs (Structured by Sprints 1, 2, 3) ----------------- #
    tab_rice, tab_roadmap, tab_dependencies, tab_insights, tab_trends, tab_segments, tab_clusters, tab_copilot = st.tabs([
        "🎯 Smart RICE & Weights",
        "🗺️ Interactive Roadmap",
        "🔗 Dependency Mapping",
        "💡 Evidence Insights",
        "📈 Trends & Forecasting",
        "👥 Customer Segments",
        "🔍 Cluster & Duplicates",
        "🤖 AI Product Copilot"
    ])

    # ----------------- SPRINT 1: SMART RICE PRIORITIZATION & CUSTOM WEIGHTS ----------------- #
    with tab_rice:
        st.subheader("🎯 Smart Priority Score (RICE) & Custom Weighting")
        st.markdown(
            "Calculate priority scores using the formula: "
            "$$\\text{Weighted RICE Score} = \\frac{(\\text{Reach} \\times W_R) \\times (\\text{Impact} \\times W_I) \\times (\\text{Confidence} \\times W_C)}{\\text{Effort} \\times W_E}$$"
        )
        
        # Custom Weights Panel
        with st.expander("⚖️ Customize Scoring Weights & Strategic Presets", expanded=True):
            col_preset, col_w_reach, col_w_impact, col_w_conf, col_w_eff = st.columns(5)
            
            with col_preset:
                preset_name = st.selectbox(
                    "Strategic Preset",
                    list(SCORING_PRESETS.keys()) + ["Custom Weights"],
                    index=list(SCORING_PRESETS.keys()).index(st.session_state.active_preset) if st.session_state.active_preset in SCORING_PRESETS else 0
                )
                if preset_name != "Custom Weights" and preset_name != st.session_state.active_preset:
                    st.session_state.active_preset = preset_name
                    p = SCORING_PRESETS[preset_name]
                    st.session_state.weight_reach = p["weight_reach"]
                    st.session_state.weight_impact = p["weight_impact"]
                    st.session_state.weight_confidence = p["weight_confidence"]
                    st.session_state.weight_effort = p["weight_effort"]
                    st.rerun()

            with col_w_reach:
                w_reach = st.slider("Reach Weight ($W_R$)", 0.2, 2.5, float(st.session_state.weight_reach), step=0.1)
            with col_w_impact:
                w_impact = st.slider("Impact Weight ($W_I$)", 0.2, 2.5, float(st.session_state.weight_impact), step=0.1)
            with col_w_conf:
                w_conf = st.slider("Confidence Weight ($W_C$)", 0.2, 2.5, float(st.session_state.weight_confidence), step=0.1)
            with col_w_eff:
                w_eff = st.slider("Effort Penalty ($W_E$)", 0.2, 2.5, float(st.session_state.weight_effort), step=0.1)

            # Check if weights changed
            if (w_reach != st.session_state.weight_reach or 
                w_impact != st.session_state.weight_impact or 
                w_conf != st.session_state.weight_confidence or 
                w_eff != st.session_state.weight_effort):
                st.session_state.weight_reach = w_reach
                st.session_state.weight_impact = w_impact
                st.session_state.weight_confidence = w_conf
                st.session_state.weight_effort = w_eff
                st.session_state.active_preset = "Custom Weights"
                
                # Recalculate ranking and roadmap
                ranking_df = rank_features(
                    df, weight_reach=w_reach, weight_impact=w_impact, 
                    weight_confidence=w_conf, weight_effort=w_eff
                )
                st.session_state.ranking_df = ranking_df
                roadmap_df = generate_ai_roadmap(df, ranking_df, st.session_state.cluster_keywords, st.session_state.cluster_summaries)
                st.session_state.roadmap_df = roadmap_df
                st.session_state.dependency_df = map_feature_dependencies(roadmap_df)
                st.rerun()

        # Interactive Matrix & Effort Adjusters
        col_matrix, col_sliders = st.columns([7, 5])
        with col_matrix:
            fig_matrix = plot_interactive_rice_matrix(roadmap_df)
            st.plotly_chart(fig_matrix, use_container_width=True)
            
        with col_sliders:
            st.markdown("#### ⚙️ Fine-Tune Engineering Effort (1 - 10)")
            st.caption("Adjust estimated effort to recalculate RICE score in real time:")
            
            recalculate = False
            for idx in range(min(7, len(roadmap_df))):
                row = roadmap_df.iloc[idx]
                theme = row['Theme']
                curr_eff = float(row['Effort'])
                new_eff = st.slider(f"{theme}", 1.0, 10.0, curr_eff, step=0.5, key=f"eff_slider_{idx}")
                if new_eff != curr_eff:
                    roadmap_df.at[roadmap_df.index[idx], 'Effort'] = new_eff
                    c_id = row['ID']
                    reach = ranking_df.loc[ranking_df['Cluster'] == c_id, 'Reach'].values[0]
                    impact = ranking_df.loc[ranking_df['Cluster'] == c_id, 'Impact'].values[0]
                    conf = ranking_df.loc[ranking_df['Cluster'] == c_id, 'Confidence'].values[0]
                    new_score = calculate_rice_score(
                        reach, impact, conf, new_eff,
                        st.session_state.weight_reach, st.session_state.weight_impact,
                        st.session_state.weight_confidence, st.session_state.weight_effort
                    )
                    ranking_df.loc[ranking_df['Cluster'] == c_id, 'RICE Score'] = new_score
                    roadmap_df.at[roadmap_df.index[idx], 'Score'] = new_score
                    roadmap_df.at[roadmap_df.index[idx], 'RICE_Score'] = new_score
                    recalculate = True
                    
            if recalculate:
                roadmap_df = roadmap_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
                st.session_state.roadmap_df = roadmap_df
                st.session_state.dependency_df = map_feature_dependencies(roadmap_df)

        st.markdown("---")
        st.markdown("### 📋 Transparent RICE Ranking Table")
        
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

    # ----------------- SPRINT 2: INTERACTIVE ROADMAP BUILDER ----------------- #
    with tab_roadmap:
        st.subheader("🗺️ Interactive Roadmap Builder (Now / Next / Later)")
        st.markdown(
            "Move features across execution horizons, reorder priorities, and export the live roadmap."
        )
        
        # Export roadmap buttons
        col_export_md, col_export_csv = st.columns([1, 4])
        with col_export_md:
            roadmap_md = export_roadmap_markdown(roadmap_df)
            st.download_button(
                "📥 Export Roadmap (.md)",
                data=roadmap_md,
                file_name="product_roadmap_now_next_later.md",
                mime="text/markdown"
            )
            
        col_now, col_next, col_later = st.columns(3)
        
        now_items = roadmap_df[roadmap_df['Stage'] == 'NOW']
        next_items = roadmap_df[roadmap_df['Stage'] == 'NEXT']
        later_items = roadmap_df[roadmap_df['Stage'] == 'LATER']
        
        with col_now:
            st.markdown(f"### 🔥 NOW <span class='badge-now'>{len(now_items)} Items</span>", unsafe_allow_html=True)
            st.caption("Immediate Sprint / Q1 • Critical Friction & High ROI")
            for idx, item in now_items.iterrows():
                with st.expander(f"**{item['Theme']}** (RICE: {item['Score']})", expanded=True):
                    st.markdown(f"**Target:** `{item['Timeframe']}` | **Team:** `{item['Assigned Team']}`")
                    st.markdown(f"**Effort:** `{item['Effort']} pts` | **Volume:** `{item['Volume']} reqs`")
                    st.info(item['Summary'])
                    
                    # Stage Move Dropdown
                    new_stg = st.selectbox(
                        "Move Horizon:", ["NOW", "NEXT", "LATER"], index=0, key=f"mv_now_{item['ID']}"
                    )
                    if new_stg != "NOW":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                    
        with col_next:
            st.markdown(f"### ⭐ NEXT <span class='badge-next'>{len(next_items)} Items</span>", unsafe_allow_html=True)
            st.caption("Upcoming Quarter / Q2 • Strategic Growth & Differentiators")
            for idx, item in next_items.iterrows():
                with st.expander(f"**{item['Theme']}** (RICE: {item['Score']})", expanded=False):
                    st.markdown(f"**Target:** `{item['Timeframe']}` | **Team:** `{item['Assigned Team']}`")
                    st.markdown(f"**Effort:** `{item['Effort']} pts` | **Volume:** `{item['Volume']} reqs`")
                    st.write(item['Summary'])
                    
                    # Stage Move Dropdown
                    new_stg = st.selectbox(
                        "Move Horizon:", ["NOW", "NEXT", "LATER"], index=1, key=f"mv_nxt_{item['ID']}"
                    )
                    if new_stg != "NEXT":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()
                    
        with col_later:
            st.markdown(f"### 🧊 LATER <span class='badge-later'>{len(later_items)} Items</span>", unsafe_allow_html=True)
            st.caption("Horizon 3 / Backlog • High Effort or Exploratory")
            for idx, item in later_items.iterrows():
                with st.expander(f"**{item['Theme']}** (RICE: {item['Score']})", expanded=False):
                    st.markdown(f"**Target:** `{item['Timeframe']}` | **Team:** `{item['Assigned Team']}`")
                    st.markdown(f"**Effort:** `{item['Effort']} pts` | **Volume:** `{item['Volume']} reqs`")
                    st.write(item['Summary'])
                    
                    # Stage Move Dropdown
                    new_stg = st.selectbox(
                        "Move Horizon:", ["NOW", "NEXT", "LATER"], index=2, key=f"mv_ltr_{item['ID']}"
                    )
                    if new_stg != "LATER":
                        st.session_state.roadmap_df = move_roadmap_item(roadmap_df, item['ID'], new_stg)
                        st.session_state.dependency_df = map_feature_dependencies(st.session_state.roadmap_df)
                        st.rerun()

    # ----------------- SPRINT 2: DEPENDENCY MAPPING ----------------- #
    with tab_dependencies:
        st.subheader("🔗 Architectural Dependency Mapping & Blocker Detection")
        st.markdown(
            "Visualize technical prerequisites and downstream feature enablers. "
            "Detect execution order violations before committing engineering capacity."
        )
        
        # Blocker warnings
        blocked_items = dependency_df[dependency_df['Is_Blocked']]
        if not blocked_items.empty:
            for _, b_row in blocked_items.iterrows():
                st.warning(
                    f"⚠️ **Roadmap Blocker:** Feature **'{b_row['Theme']}'** is scheduled in **{b_row['Current_Stage']}**, "
                    f"but {b_row['Dependency_Status'].replace('⚠️ Blocked by ', 'its prerequisite ')}! "
                    f"Move the prerequisite forward or defer this item."
                )
        else:
            st.success("✅ **Roadmap Validated:** All scheduled features have their foundational prerequisites satisfied.")
            
        st.markdown("### 🗺️ Full Feature Dependency Matrix")
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

    # ----------------- EVIDENCE-BACKED AI INSIGHTS ----------------- #
    with tab_insights:
        st.subheader("💡 Evidence-Backed AI Recommendations")
        st.markdown(
            "Every AI recommendation includes transparent rationale, supporting demand metrics, customer segment breakdown, "
            "and **verbatim user quotes** to prevent unsupported AI claims."
        )
        
        for ins in insights:
            with st.container():
                st.markdown(f"### {ins['Rationale_Badge']}: {ins['Theme']}")
                
                col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
                with col_meta1:
                    st.metric("RICE Priority", f"{ins['RICE_Score']}")
                with col_meta2:
                    st.metric("Supporting Demand", f"{ins['Request_Count']} requests", f"{ins['Pct_Of_Total']}% share")
                with col_meta3:
                    st.metric("Total Upvotes", f"{ins['Total_Upvotes']:,}")
                with col_meta4:
                    st.metric("Customer Friction", f"{ins['Sentiment_Breakdown']['Negative_Pct']}% negative")
                    
                st.markdown(f"**🤖 Why the AI made this recommendation:** {ins['AI_Rationale']}")
                st.markdown(f"**🎯 Actionable PM Recommendation:** `{ins['Actionable_Recommendation']}`")
                st.markdown(f"**👥 Segment Concentration:** {ins['Segment_Summary']}")
                
                st.markdown("##### 💬 Verbatim Customer Feedback Samples:")
                for q in ins['Customer_Quotes']:
                    st.markdown(
                        f"""<div class="quote-card">
                            "{q['text']}" 
                            <br><small style="color: #64748b;">— <strong>{q['segment']}</strong> Customer via {q['source']} ({q['upvotes']} upvotes, {q['date']})</small>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                st.markdown("---")

    # ----------------- SPRINT 3: DEMAND TRENDS & FORECASTING ----------------- #
    with tab_trends:
        st.subheader("📈 Demand Trend Forecasting & Momentum")
        st.markdown(
            "Track customer interest velocity over time to catch emerging features **before they become the most requested**."
        )
        
        if trends_df is not None and not trends_df.empty:
            # Interactive monthly trend
            fig_trends = plot_interactive_trends(filtered, trends_df)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            st.markdown("### 🚀 Momentum Tracker (Velocity in Recent 60 Days)")
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

    # ----------------- SPRINT 3: CUSTOMER SEGMENT ANALYSIS ----------------- #
    with tab_segments:
        st.subheader("👥 Customer Segment Breakdown")
        st.markdown(
            "Understand how priorities differ between **Enterprise, Pro, Starter, and Free** customer tiers."
        )
        
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
        st.markdown("### 🏆 Top Requested Features by Customer Segment")
        
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

    # ----------------- CLUSTER & DUPLICATE EXPLORER ----------------- #
    with tab_clusters:
        st.subheader("🔍 Semantic Duplicate Detection & Cluster Explorer")
        st.markdown(
            "FAISS vector similarity groups redundant requests to prevent duplicate backlog tickets and highlight common patterns."
        )
        
        # Duplicate groups expander
        st.markdown(f"#### 🔗 Duplicate Groups ({len(st.session_state.duplicate_groups or [])} groups identified)")
        if st.session_state.duplicate_groups:
            for g in (st.session_state.duplicate_groups[:8]):
                prim_row = df.iloc[g['primary_idx']]
                with st.expander(f"📌 Primary: \"{prim_row['Request_Text']}\" ({g['count']} identical/near duplicates)"):
                    st.caption(f"Semantic similarity: **{g['avg_similarity']}** | Category: **{prim_row['Category']}**")
                    st.markdown("**Duplicate Requests Merged:**")
                    for d_idx in g['duplicate_indices']:
                        dup_row = df.iloc[d_idx]
                        st.markdown(f"- \"{dup_row['Request_Text']}\" *({dup_row['Customer_Segment']} tier, {dup_row['Upvotes']} upvotes)*")
        else:
            st.info("No semantic duplicates found above the current similarity threshold.")
            
        st.markdown("---")
        st.markdown("#### ☁️ Customer Feedback Keyword Cloud")
        all_feedback_text = " ".join(filtered['Cleaned_Text'].tolist()[:300])
        fig_wc = generate_wordcloud(all_feedback_text)
        st.pyplot(fig_wc)

    # ----------------- AI PRODUCT COPILOT ----------------- #
    with tab_copilot:
        st.subheader("🤖 AI Product Copilot & Generation")
        
        copilot_sub1, copilot_sub2, copilot_sub3 = st.tabs([
            "🔎 Semantic Vector Search",
            "📝 AI PRD Generator",
            "📣 Release Notes Generator"
        ])
        
        # 1. Semantic Search
        with copilot_sub1:
            st.markdown("#### Instant Semantic Search")
            st.caption("Ask natural language queries like *'Why are users complaining about dark mode or logins?'*:")
            
            search_query = st.text_input("Enter search query:", placeholder="E.g., SSO authentication, export to excel, tablet layout")
            if search_query:
                embed_model = load_embedding_model()
                q_emb = embed_model.encode([search_query])[0]
                sims, indices = semantic_search(q_emb, st.session_state.index, k=5)
                
                st.markdown(f"**Found top matches for '{search_query}':**")
                for sim, idx in zip(sims, indices):
                    if idx < len(df):
                        match_row = df.iloc[idx]
                        st.markdown(f"""
                        <div class="quote-card">
                            "{match_row['Request_Text']}"
                            <br><small><strong>Category:</strong> {match_row['Category']} | <strong>Similarity:</strong> {sim:.2f} | <strong>Sentiment:</strong> {match_row['Sentiment']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
        # 2. AI PRD Generator
        with copilot_sub2:
            st.markdown("#### Automated Product Requirements Document (PRD) Generator")
            st.caption("Select any feature theme to synthesize customer feedback into a ready-to-use PRD:")
            
            theme_choice = st.selectbox("Select Theme to Generate PRD:", roadmap_df['Theme'].tolist(), key="prd_theme_select")
            if st.button("✨ Generate PRD", key="gen_prd_btn"):
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
                    
        # 3. Release Notes Generator
        with copilot_sub3:
            st.markdown("#### Automated Customer-Facing Release Notes Generator")
            st.caption("Turn planned features into engaging, public product marketing announcements:")
            
            rn_choice = st.selectbox("Select Feature Theme:", roadmap_df['Theme'].tolist(), key="rn_theme_select")
            if st.button("✨ Draft Release Notes", key="gen_rn_btn"):
                with st.spinner("Drafting release notes..."):
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
    # ----------------- Clean Empty State ----------------- #
    st.markdown("""
    <div class="saas-card" style="text-align: center; padding: 60px 20px;">
        <div style="font-size: 3.5rem; margin-bottom: 12px;">📊</div>
        <h2 style="font-weight: 800; margin-bottom: 8px;">No Feedback Data Loaded Yet</h2>
        <p style="color: #64748b; max-width: 550px; margin: 0 auto 24px auto;">
            Get started by loading the <strong>Sample Data (1,000 requests)</strong> in the left sidebar, 
            or upload your own customer feedback CSV.
        </p>
        <p style="color: #94a3b8; font-size: 0.85rem;">
            Supported columns: <code>Request_Text</code>, <code>Category</code>, <code>Upvotes</code>, <code>Date</code>, <code>Customer_Segment</code>, <code>Source</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
