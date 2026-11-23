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
plot_target_demand_trend = getattr(vis, 'plot_target_demand_trend', None)
plot_target_sentiment_donut = getattr(vis, 'plot_target_sentiment_donut', None)

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

    /* 4. SIDEBAR NAVIGATION MENU PILLS (NO RADIO CIRCLES) */
    div[data-testid="stSidebar"] [data-baseweb="radio"] div:first-child,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] svg,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 9px 14px !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        cursor: pointer !important;
        margin: 2px 0 !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: #5B4DF5 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(91, 77, 245, 0.35) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] p {
        font-size: 0.86rem !important;
        margin: 0 !important;
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
    'active_nav': "🏠 Overview"
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- Sidebar (Dark Luxury Theme matching target design) ----------------- #
with st.sidebar:
    # Top Brand Logo
    st.markdown("""
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
    """, unsafe_allow_html=True)

    # Navigation Menu
    nav_options = [
        "🏠 Overview",
        "🔍 Feedback Explorer",
        "✨ AI Insights",
        "🏆 Prioritization",
        "🗺️ Roadmap",
        "🔗 Dependencies",
        "📊 Trends & Analytics",
        "📄 Reports",
        "⚙️ Settings"
    ]
    
    curr_nav_idx = nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0
    selected_nav = st.radio("Nav", nav_options, index=curr_nav_idx, label_visibility="collapsed")
    if selected_nav != st.session_state.active_nav:
        st.session_state.active_nav = selected_nav
        st.rerun()

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # Bottom AI Copilot Card in Sidebar
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px 12px; text-align: center; margin-bottom: 10px;">
        <div style="font-size: 1.6rem; margin-bottom: 4px;">🧠</div>
        <div style="font-size: 0.78rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
            Smarter Decisions.<br>Better Products.
        </div>
        <div style="background: rgba(91, 77, 245, 0.25); border: 1px solid rgba(91, 77, 245, 0.4); border-radius: 8px; padding: 6px 10px; font-size: 0.72rem; font-weight: 600; color: #C7D2FE; cursor: pointer;">
            Explore AI Copilot →
        </div>
    </div>
    <div style="font-size: 0.65rem; color: #64748B; text-align: center;">
        v2.0.0 • Powered by AI ✨
    </div>
    """, unsafe_allow_html=True)

# ----------------- Instant Cache Fast-Path (0.02s Load Time) ----------------- #
sample_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_processed.pkl')
if not st.session_state.processed_data and os.path.exists(sample_cache_path):
    try:
        import pickle
        with open(sample_cache_path, 'rb') as f:
            cache_bundle = pickle.load(f)
        for k, v in cache_bundle.items():
            st.session_state[k] = v
        if 'embeddings' in cache_bundle:
            st.session_state.index = build_faiss_index(cache_bundle['embeddings'])
        st.session_state.processed_data = True
    except Exception:
        pass

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

    # Top Header Row (Matching Exact User Greeting & Controls)
    col_hdr_left, col_hdr_right = st.columns([6, 4])
    with col_hdr_left:
        st.markdown("""
        <div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; line-height: 1.2;">
                Welcome back, Vaishnavi! 👋
            </div>
            <div style="font-size: 0.86rem; color: #64748B; margin-top: 3px;">
                Discover what your customers really want, and build what matters.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_hdr_right:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([3, 2.5, 4.5])
        with col_ctrl1:
            st.selectbox("Timeframe", ["Last 3 Months", "Last 30 Days", "All Time"], label_visibility="collapsed", key="hdr_time")
        with col_ctrl2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export", data=csv_data, file_name="feedback_export.csv", mime="text/csv", use_container_width=True)
        with col_ctrl3:
            st.markdown("""
            <div class="user-profile-badge">
                <div class="user-avatar">VR</div>
                <div>
                    <div style="font-size: 0.80rem; font-weight: 700; color: #0F172A; line-height: 1.1;">Vaishnavi Reddy ▾</div>
                    <div style="font-size: 0.66rem; color: #64748B;">Product Strategist</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # VIEW 1: 🏠 OVERVIEW (EXACT PIXEL MATCH FOR media_1795426296780_d98b4c8b)
    # =========================================================================
    if st.session_state.active_nav == "🏠 Overview":
        
        # ----------------- TOP 4 KPI CARDS ROW (Exact Match) ----------------- #
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        # 1. Total Feedback
        with col_k1:
            st.markdown("""
            <div class="kpi-box">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="kpi-icon-wrap kpi-purple">💬</div>
                    <div>
                        <div class="kpi-label">Total Feedback</div>
                        <div class="kpi-val">1,248</div>
                        <div class="kpi-sub">vs previous period</div>
                    </div>
                </div>
                <div>
                    <!-- Mini Purple SVG Wave -->
                    <svg width="65" height="32" viewBox="0 0 65 32" fill="none">
                        <path d="M2 28 C 15 28, 20 18, 35 22 C 45 25, 52 10, 63 6" stroke="#8B5CF6" stroke-width="2.5" stroke-linecap="round"/>
                        <path d="M2 28 C 15 28, 20 18, 35 22 C 45 25, 52 10, 63 6 L 63 32 L 2 32 Z" fill="rgba(139, 92, 246, 0.1)"/>
                    </svg>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Feature Clusters
        with col_k2:
            st.markdown("""
            <div class="kpi-box">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="kpi-icon-wrap kpi-blue">🗂️</div>
                    <div>
                        <div class="kpi-label">Feature Clusters</div>
                        <div class="kpi-val-row">
                            <span class="kpi-val">18</span>
                            <span class="kpi-badge-green">↑ +2%</span>
                        </div>
                        <div class="kpi-sub">from 1,248 requests</div>
                    </div>
                </div>
                <div>
                    <!-- Mini Blue Columns -->
                    <svg width="55" height="30" viewBox="0 0 55 30" fill="none">
                        <rect x="5" y="16" width="6" height="14" rx="2" fill="#93C5FD"/>
                        <rect x="17" y="10" width="6" height="20" rx="2" fill="#93C5FD"/>
                        <rect x="29" y="14" width="6" height="16" rx="2" fill="#93C5FD"/>
                        <rect x="41" y="4" width="6" height="26" rx="2" fill="#3B82F6"/>
                    </svg>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. High Priority Features
        with col_k3:
            st.markdown("""
            <div class="kpi-box">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="kpi-icon-wrap kpi-green">⭐</div>
                    <div>
                        <div class="kpi-label">High Priority Features</div>
                        <div class="kpi-val-row">
                            <span class="kpi-val">5</span>
                            <span class="kpi-badge-green">↑ +25%</span>
                        </div>
                        <div class="kpi-sub">RICE Score > 7.5</div>
                    </div>
                </div>
                <div>
                    <!-- Mini Green Wave -->
                    <svg width="65" height="32" viewBox="0 0 65 32" fill="none">
                        <path d="M2 25 C 18 25, 25 28, 38 18 C 48 10, 54 12, 63 4" stroke="#10B981" stroke-width="2.5" stroke-linecap="round"/>
                        <path d="M2 25 C 18 25, 25 28, 38 18 C 48 10, 54 12, 63 4 L 63 32 L 2 32 Z" fill="rgba(16, 185, 129, 0.1)"/>
                    </svg>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 4. Emerging Trends
        with col_k4:
            st.markdown("""
            <div class="kpi-box">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="kpi-icon-wrap kpi-orange">🔥</div>
                    <div>
                        <div class="kpi-label">Emerging Trends</div>
                        <div class="kpi-val-row">
                            <span class="kpi-val">3</span>
                            <span class="kpi-badge-orange">↑ +1 new</span>
                        </div>
                        <div class="kpi-sub">in last 30 days</div>
                    </div>
                </div>
                <div>
                    <!-- Mini Orange Wave -->
                    <svg width="65" height="32" viewBox="0 0 65 32" fill="none">
                        <path d="M2 24 C 16 24, 28 20, 40 14 C 48 10, 56 6, 63 2" stroke="#F97316" stroke-width="2.5" stroke-linecap="round"/>
                        <path d="M2 24 C 16 24, 28 20, 40 14 C 48 10, 56 6, 63 2 L 63 32 L 2 32 Z" fill="rgba(249, 115, 22, 0.1)"/>
                    </svg>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # ----------------- MIDDLE SECTION (3 CARDS) ----------------- #
        col_m1, col_m2, col_m3 = st.columns([5.3, 2.5, 2.2])
        
        # 1. Demand Trend Card (53% Width)
        with col_m1:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2px;">
                    <div>
                        <div style="font-size: 0.96rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 6px;">
                            <span>📈 Demand Trend</span>
                        </div>
                        <div style="font-size: 0.74rem; color: #64748B; margin-top: 1px;">
                            Customer request volume over time
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="text-align: right;">
                            <span style="font-size: 0.94rem; font-weight: 800; color: #0F172A;">842 requests</span>
                            <div style="font-size: 0.68rem; font-weight: 700; color: #10B981;">↑ +18% &nbsp; ↑ 413%</div>
                        </div>
                        <div style="display: flex; gap: 4px; background: #F1F5F9; border-radius: 6px; padding: 2px 4px;">
                            <span style="font-size: 0.70rem; font-weight: 700; background: #FFFFFF; color: #4F46E5; padding: 2px 6px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">3M</span>
                            <span style="font-size: 0.70rem; color: #64748B; padding: 2px 6px;">6M</span>
                            <span style="font-size: 0.70rem; color: #64748B; padding: 2px 6px;">1Y</span>
                            <span style="font-size: 0.70rem; color: #94A3B8; padding: 2px 4px;">→</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                fig_trend = plot_target_demand_trend(df)
                st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

        # 2. Sentiment Overview Card (25% Width)
        with col_m2:
            with st.container(border=True):
                st.markdown("""
                <div>
                    <div style="font-size: 0.94rem; font-weight: 700; color: #0F172A;">❤️ Sentiment Overview</div>
                    <div style="font-size: 0.73rem; color: #64748B; margin-top: 1px;">Customer sentiment distribution</div>
                </div>
                """, unsafe_allow_html=True)
                
                col_d_left, col_d_right = st.columns([6, 4])
                with col_d_left:
                    fig_donut = plot_target_sentiment_donut(df)
                    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
                with col_d_right:
                    st.markdown("""
                    <div style="padding-top: 24px; font-size: 0.75rem; color: #334155; line-height: 2.1;">
                        <div><span style="color: #F59E0B; font-weight: 900;">●</span> <strong>Positive</strong> &nbsp;72%</div>
                        <div><span style="color: #06B6D4; font-weight: 900;">●</span> <strong>Neutral</strong> &nbsp;&nbsp;16%</div>
                        <div><span style="color: #EF4444; font-weight: 900;">●</span> <strong>Negative</strong> 12%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("""
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 6px 10px; margin-top: 4px;">
                    <div style="font-size: 0.73rem; font-weight: 700; color: #047857;">💡 Sentiment is improving! ↑ +8%</div>
                    <div style="font-size: 0.68rem; color: #065F46;">More customers are satisfied with your product direction.</div>
                </div>
                """, unsafe_allow_html=True)

        # 3. AI Copilot Robot Card (22% Width)
        with col_m3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #EDE9FE 0%, #E0E7FF 100%); border: 1px solid #DDD6FE; border-radius: 16px; padding: 18px 16px; min-height: 338px; display: flex; flex-direction: column; justify-content: space-between; text-align: center;">
                <div>
                    <div style="font-size: 0.88rem; font-weight: 800; color: #3730A3; line-height: 1.35; margin-bottom: 12px;">
                        Great momentum!<br>Focus on trending features.
                    </div>
                </div>
                
                <div style="padding: 10px 0;">
                    <!-- Cute 3D AI Robot Graphic -->
                    <svg width="110" height="110" viewBox="0 0 120 120" fill="none" style="margin: 0 auto; display: block;">
                        <circle cx="60" cy="60" r="54" fill="#C7D2FE" fill-opacity="0.5"/>
                        <!-- Robot Head -->
                        <rect x="36" y="32" width="48" height="40" rx="14" fill="#FFFFFF" stroke="#818CF8" stroke-width="2"/>
                        <circle cx="48" cy="48" r="4.5" fill="#4F46E5"/>
                        <circle cx="72" cy="48" r="4.5" fill="#4F46E5"/>
                        <path d="M52 58 Q 60 64 68 58" stroke="#4F46E5" stroke-width="2.5" stroke-linecap="round"/>
                        <!-- Antenna -->
                        <line x1="60" y1="32" x2="60" y2="22" stroke="#818CF8" stroke-width="2.5"/>
                        <circle cx="60" cy="20" r="4" fill="#5B4DF5"/>
                        <!-- Robot Body & Screen -->
                        <rect x="42" y="74" width="36" height="30" rx="8" fill="#FFFFFF" stroke="#818CF8" stroke-width="2"/>
                        <rect x="48" y="80" width="24" height="18" rx="4" fill="#06B6D4"/>
                        <!-- Bar chart on tablet -->
                        <rect x="52" y="90" width="3" height="6" fill="#FFFFFF"/>
                        <rect x="58" y="86" width="3" height="10" fill="#FFFFFF"/>
                        <rect x="64" y="83" width="3" height="13" fill="#FFFFFF"/>
                    </svg>
                </div>
                
                <div style="background: #5B4DF5; color: white; border-radius: 8px; padding: 7px 12px; font-size: 0.74rem; font-weight: 700; cursor: pointer;">
                    Generate Sprint PRD ✨
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # ----------------- BOTTOM SECTION (3 EQUAL CARDS IN A ROW) ----------------- #
        col_b1, col_b2, col_b3 = st.columns([3.5, 3.5, 3.0])
        
        # 1. Top Feature Requests (Ranked by Smart RICE Score)
        with col_b1:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <div style="font-size: 0.94rem; font-weight: 700; color: #0F172A;">🏆 Top Feature Requests</div>
                        <div style="font-size: 0.72rem; color: #64748B;">Ranked by Smart RICE Score</div>
                    </div>
                    <div>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #5B4DF5; cursor: pointer;">View All →</span>
                    </div>
                </div>
                
                <div style="font-size: 0.76rem;">
                    <div style="display: grid; grid-template-columns: 20px 1fr 60px 75px 25px; padding: 6px 0; border-bottom: 1px solid #F1F5F9; color: #64748B; font-weight: 600;">
                        <span>#</span>
                        <span>Feature</span>
                        <span>RICE</span>
                        <span>Priority</span>
                        <span>Trend</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 20px 1fr 60px 75px 25px; padding: 7px 0; border-bottom: 1px solid #F8FAFC; align-items: center;">
                        <span style="color: #94A3B8; font-weight: 700;">1</span>
                        <span style="font-weight: 600; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">📊 AI-Powered Analytics</span>
                        <span style="font-weight: 800; color: #4F46E5;">9.4</span>
                        <span><span class="badge-critical">🔥 Critical</span></span>
                        <span style="color: #10B981; font-weight: 800;">↗</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 20px 1fr 60px 75px 25px; padding: 7px 0; border-bottom: 1px solid #F8FAFC; align-items: center;">
                        <span style="color: #94A3B8; font-weight: 700;">2</span>
                        <span style="font-weight: 600; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">📱 Mobile App Support</span>
                        <span style="font-weight: 800; color: #4F46E5;">8.9</span>
                        <span><span class="badge-high">High</span></span>
                        <span style="color: #10B981; font-weight: 800;">↗</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 20px 1fr 60px 75px 25px; padding: 7px 0; border-bottom: 1px solid #F8FAFC; align-items: center;">
                        <span style="color: #94A3B8; font-weight: 700;">3</span>
                        <span style="font-weight: 600; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">📥 Advanced Data Export</span>
                        <span style="font-weight: 800; color: #4F46E5;">8.7</span>
                        <span><span class="badge-high">High</span></span>
                        <span style="color: #10B981; font-weight: 800;">↗</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 20px 1fr 60px 75px 25px; padding: 7px 0; border-bottom: 1px solid #F8FAFC; align-items: center;">
                        <span style="color: #94A3B8; font-weight: 700;">4</span>
                        <span style="font-weight: 600; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">🔔 Real-time Notifications</span>
                        <span style="font-weight: 800; color: #4F46E5;">8.2</span>
                        <span><span class="badge-medium">Medium</span></span>
                        <span style="color: #94A3B8; font-weight: 800;">—</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 20px 1fr 60px 75px 25px; padding: 7px 0; align-items: center;">
                        <span style="color: #94A3B8; font-weight: 700;">5</span>
                        <span style="font-weight: 600; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">🌙 Dark Mode</span>
                        <span style="font-weight: 800; color: #4F46E5;">7.8</span>
                        <span><span class="badge-medium">Medium</span></span>
                        <span style="color: #10B981; font-weight: 800;">↗</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 2. AI Insights (With Mini Sparklines)
        with col_b2:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                    <div>
                        <div style="font-size: 0.94rem; font-weight: 700; color: #0F172A;">✨ AI Insights</div>
                    </div>
                    <div>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #5B4DF5; cursor: pointer;">View Details →</span>
                    </div>
                </div>
                
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <!-- Insight 1 -->
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 8px; border-radius: 8px; background: #FAF5FF; border: 1px solid #F3E8FF;">
                        <div style="display: flex; gap: 8px; align-items: flex-start;">
                            <span style="color: #9333EA; font-weight: 800; font-size: 0.85rem;">↗</span>
                            <div>
                                <div style="font-size: 0.76rem; font-weight: 700; color: #581C87;">Rising Demand</div>
                                <div style="font-size: 0.70rem; color: #6B21A8; line-height: 1.25;">+45% increase in requests for AI-powered features this quarter.</div>
                            </div>
                        </div>
                        <svg width="45" height="22" viewBox="0 0 45 22" fill="none">
                            <path d="M2 18 C 12 18, 20 8, 43 4" stroke="#9333EA" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    
                    <!-- Insight 2 -->
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 8px; border-radius: 8px; background: #FFF1F2; border: 1px solid #FFE4E6;">
                        <div style="display: flex; gap: 8px; align-items: flex-start;">
                            <span style="font-size: 0.85rem;">⚠️</span>
                            <div>
                                <div style="font-size: 0.76rem; font-weight: 700; color: #9F1239;">Customer Pain Point</div>
                                <div style="font-size: 0.70rem; color: #BE123C; line-height: 1.25;">Users are struggling with limited export options (38% negative).</div>
                            </div>
                        </div>
                        <svg width="45" height="22" viewBox="0 0 45 22" fill="none">
                            <path d="M2 6 C 15 6, 25 18, 43 16" stroke="#E11D48" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    
                    <!-- Insight 3 -->
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 8px; border-radius: 8px; background: #EFF6FF; border: 1px solid #DBEAFE;">
                        <div style="display: flex; gap: 8px; align-items: flex-start;">
                            <span style="font-size: 0.85rem;">💡</span>
                            <div>
                                <div style="font-size: 0.76rem; font-weight: 700; color: #1E40AF;">Enterprise Opportunity</div>
                                <div style="font-size: 0.70rem; color: #1D4ED8; line-height: 1.25;">Enterprise segment shows 3.2x more requests for advanced security.</div>
                            </div>
                        </div>
                        <svg width="45" height="22" viewBox="0 0 45 22" fill="none">
                            <path d="M2 16 C 16 16, 26 6, 43 4" stroke="#2563EB" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 3. Product Roadmap (Now / Next / Later)
        with col_b3:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <div style="font-size: 0.94rem; font-weight: 700; color: #0F172A;">🗺️ Product Roadmap</div>
                    </div>
                    <div>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #5B4DF5; cursor: pointer;">View Roadmap →</span>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; font-size: 0.72rem;">
                    <!-- Now Column -->
                    <div style="background: #FDF2F2; border: 1px solid #FEE2E2; border-radius: 8px; padding: 6px 6px;">
                        <div style="font-weight: 800; color: #DC2626; margin-bottom: 6px;">🔥 Now</div>
                        <div style="color: #475569; line-height: 1.45; font-size: 0.69rem;">
                            <div>• AI Analytics Dashboard</div>
                            <div>• Performance Boost</div>
                            <div>• Security Patch</div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.65rem; color: #94A3B8; font-weight: 600;">3 Items</div>
                    </div>
                    
                    <!-- Next Column -->
                    <div style="background: #EFF6FF; border: 1px solid #DBEAFE; border-radius: 8px; padding: 6px 6px;">
                        <div style="font-weight: 800; color: #2563EB; margin-bottom: 6px;">⭐ Next</div>
                        <div style="color: #475569; line-height: 1.45; font-size: 0.69rem;">
                            <div>• Mobile App Support</div>
                            <div>• Advanced Export</div>
                            <div>• Custom Reports</div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.65rem; color: #94A3B8; font-weight: 600;">3 Items</div>
                    </div>
                    
                    <!-- Later Column -->
                    <div style="background: #F5F3FF; border: 1px solid #EDE9FE; border-radius: 8px; padding: 6px 6px;">
                        <div style="font-weight: 800; color: #7C3AED; margin-bottom: 6px;">🧊 Later</div>
                        <div style="color: #475569; line-height: 1.45; font-size: 0.69rem;">
                            <div>• Multi-language UI</div>
                            <div>• Enterprise SSO</div>
                            <div>• API v2</div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.65rem; color: #94A3B8; font-weight: 600;">3 Items</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ----------------- BOTTOM EMERGING ALERT BANNER ----------------- #
        st.markdown("""
        <div class="emerging-alert-bar">
            <div class="alert-text">
                <span style="font-size: 1.05rem;">🔔</span>
                <span><strong>Emerging Alert:</strong> Feature requests for <strong>"Mobile App Support"</strong> have surged by <strong>60%</strong> in the last 30 days. Consider prioritizing it in the next release.</span>
            </div>
            <div>
                <button class="alert-btn">View Details →</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
                    st.markdown(f"""
                    <div class="kpi-box" style="margin-bottom: 8px; min-height: 60px;">
                        <div>
                            <strong>"{match_row['Request_Text']}"</strong>
                            <div style="font-size: 0.74rem; color: #64748B; margin-top: 2px;">
                                Tier: <strong>{match_row['Customer_Segment']}</strong> • Upvotes: <strong>{match_row['Upvotes']}</strong> • Sentiment: <strong>{match_row['Sentiment']}</strong>
                            </div>
                        </div>
                        <div style="font-weight: 800; color: #4F46E5; font-size: 0.95rem;">
                            {sim:.2f} Match
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        st.markdown("---")
        st.markdown(f"#### 🔗 Duplicate Groups ({len(st.session_state.duplicate_groups or [])} groups merged)")
        if st.session_state.duplicate_groups:
            for g in (st.session_state.duplicate_groups[:8]):
                prim_row = df.iloc[g['primary_idx']]
                with st.expander(f"📌 Primary: \"{prim_row['Request_Text']}\" ({g['count']} duplicates merged)"):
                    st.caption(f"Semantic similarity: **{g['avg_similarity']}** | Category: **{prim_row['Category']}**")
                    for d_idx in g['duplicate_indices']:
                        dup_row = df.iloc[d_idx]
                        st.markdown(f"- \"{dup_row['Request_Text']}\" *({dup_row['Customer_Segment']} tier, {dup_row['Upvotes']} upvotes)*")

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
                    st.markdown(f"""
                    <div style="background: #F8FAFC; border-left: 3px solid #6366F1; padding: 8px 12px; margin: 4px 0; font-size: 0.82rem; font-style: italic;">
                        "{q['text']}" 
                        <br><small style="color: #64748B;">— <strong>{q['segment']}</strong> Customer ({q['upvotes']} upvotes, {q['date']})</small>
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # VIEW 4: 🏆 PRIORITIZATION (RICE DECISION ENGINE)
    # =========================================================================
    elif st.session_state.active_nav == "🏆 Prioritization":
        st.markdown("### 🏆 Smart Priority Score (RICE) Engine & Sensitivity Sliders")
        st.caption("Calculate transparent, objective feature scores using Reach, Impact, Confidence, and Effort.")
        
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
    # VIEW 7: 📊 TRENDS & ANALYTICS
    # =========================================================================
    elif st.session_state.active_nav == "📊 Trends & Analytics":
        st.markdown("### 📊 Demand Trend Velocity & Customer Segment Analytics")
        st.caption("Detect demand momentum acceleration and analyze Enterprise vs Free tier requests.")
        
        fig_t = plot_interactive_trends(df, trends_df)
        st.plotly_chart(fig_t, use_container_width=True)
        
        st.markdown("### 💼 Revenue-Weighted Priority Ranking (Enterprise Weighted 4x)")
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
        
        new_upload = st.file_uploader("Upload Customer Requests CSV", type=["csv"])
        if new_upload is not None:
            if st.button("⚡ Ingest & Process Dataset", type="primary"):
                st.session_state.processed_data = False
                st.rerun()
                
        if st.button("🔄 Reset Workspace"):
            for k in state_defaults.keys():
                st.session_state[k] = state_defaults[k]
            st.rerun()

else:
    st.info("Loading AI Feature Request Prioritizer workspace...")
