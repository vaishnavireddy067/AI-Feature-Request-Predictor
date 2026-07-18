import streamlit as st
import pandas as pd
import os
import io
import time

# Import utils
from utils.preprocessing import load_and_clean_data
from utils.embeddings import load_embedding_model, generate_embeddings
from utils.vector_search import build_faiss_index, find_duplicates, semantic_search
from utils.clustering import perform_clustering
from utils.summarizer import summarize_cluster
from utils.keywords import load_keyword_model, extract_keywords
from utils.ranking import rank_features
from utils.visualization import plot_cluster_distribution, plot_popularity_pie, generate_wordcloud, plot_sentiment_trends, plot_impact_effort_matrix
from utils.sentiment import load_sentiment_model, analyze_sentiments
from utils.strategy import tag_dataframe

# Set page config MUST BE FIRST
st.set_page_config(
    page_title="AI Product Strategy",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium SaaS UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
 
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-color);
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Headers */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


st.title("🚀 AI Product Strategy")
st.markdown("**Enterprise Feature Prioritization & Customer Feedback Analytics**")

# Initialize session state
for key in ['processed_data', 'df', 'index', 'ranking_df', 'cluster_summaries', 'cluster_keywords', 'total_duplicates', 'messages', 'roadmap_df']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'messages' else (None if key != 'processed_data' else False)

# Sidebar setup
st.sidebar.title("⚙️ Data Pipeline")

uploaded_file = st.sidebar.file_uploader("Upload CSV Feature Requests", type="csv")

if st.sidebar.button("Load Sample Data (1000 requests)"):
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_requests.csv')
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            uploaded_file = io.StringIO(f.read())
            st.sidebar.success("Sample data loaded!")
    else:
        st.sidebar.error("Sample data not found. Run generate_sample_data.py first.")

# Processing button
if uploaded_file is not None:
    if st.sidebar.button("⚡ Process & Analyze") or not st.session_state.processed_data:
        
        progress_bar = st.progress(0, text="Initializing Data Pipeline...")
        
        start_time = time.time()
        
        progress_bar.progress(10, text="Loading and cleaning data...")
        df = load_and_clean_data(uploaded_file)
        df['Status'] = "New"
        
        progress_bar.progress(20, text="Warming up Machine Learning Models...")
        embed_model = load_embedding_model()
        kw_model = load_keyword_model()
        sentiment_model = load_sentiment_model()
        
        progress_bar.progress(35, text="Analyzing Sentiments...")
        df['Sentiment'] = analyze_sentiments(df['Cleaned_Text'].tolist(), sentiment_model)
        
        progress_bar.progress(45, text="Auto-Tagging Cross-Functional Teams...")
        df = tag_dataframe(df)
        
        progress_bar.progress(55, text="Generating Vector Embeddings...")
        embeddings = generate_embeddings(df['Cleaned_Text'].tolist(), embed_model)
        
        progress_bar.progress(65, text="Building FAISS Index & Identifying Duplicates...")
        index = build_faiss_index(embeddings)
        st.session_state.index = index
        dup_groups = find_duplicates(index, embeddings, threshold=0.2)
        st.session_state.total_duplicates = sum(len(g)-1 for g in dup_groups if len(g) > 1)
        
        progress_bar.progress(75, text="Clustering Semantic Features (KMeans)...")
        n_clusters = min(15, max(2, len(df) // 20))
        df['Cluster'] = perform_clustering(embeddings, n_clusters=n_clusters)
        
        progress_bar.progress(85, text="Extracting Keywords & Extractive Summaries...")
        cluster_summaries = {}
        cluster_keywords = {}
        for c_id in df['Cluster'].unique():
            c_texts = df[df['Cluster'] == c_id]['Cleaned_Text'].tolist()
            cluster_summaries[c_id] = summarize_cluster(c_texts, embed_model)
            cluster_keywords[c_id] = extract_keywords(" ".join(c_texts), kw_model)
            
        st.session_state.cluster_summaries = cluster_summaries
        st.session_state.cluster_keywords = cluster_keywords
        
        progress_bar.progress(95, text="Ranking & Prioritization Matrix...")
        ranking_df = rank_features(df, cluster_col='Cluster', upvotes_col='Upvotes')
        st.session_state.ranking_df = ranking_df
        
        # Pre-compute Roadmap data
        roadmap_data = []
        max_score = ranking_df['Popularity Score'].max()
        for i in range(len(ranking_df)):
            row = ranking_df.iloc[i]
            c_id = int(row['Cluster'])
            c_label = st.session_state.cluster_keywords[c_id][0].title() if st.session_state.cluster_keywords[c_id] else f"Theme {c_id}"
            
            cluster_df = df[df['Cluster'] == c_id]
            neg_ratio = len(cluster_df[cluster_df['Sentiment'] == 'NEGATIVE']) / len(cluster_df)
            score_ratio = row['Popularity Score'] / max_score
            
            # Simple AI Heuristic for Priority
            if score_ratio > 0.6 and neg_ratio > 0.4:
                priority = "🔥 NOW"
            elif score_ratio > 0.4:
                priority = "⭐ NEXT"
            else:
                priority = "🧊 LATER"
                
            # Aggregate tags
            all_tags = []
            for t in cluster_df['Auto_Tags']:
                all_tags.extend(t)
            top_tag = pd.Series(all_tags).mode()[0] if all_tags else "General"
                
            roadmap_data.append({
                "ID": c_id,
                "Theme": c_label,
                "Priority": priority,
                "Score": row['Popularity Score'],
                "Volume": row['Request Count'],
                "Assigned Team": top_tag,
                "Summary": st.session_state.cluster_summaries[c_id][:60] + "...",
                "Effort": 5  # Default effort
            })
            
        st.session_state.roadmap_df = pd.DataFrame(roadmap_data).sort_values("Score", ascending=False)
        st.session_state.df = df 
        st.session_state.processed_data = True
        
        elapsed = time.time() - start_time
        progress_bar.progress(100, text=f"✅ Analysis Complete in {elapsed:.2f} seconds!")
        time.sleep(1)
        progress_bar.empty()

# ----------------- Tabbed SaaS Layout ----------------- #
if st.session_state.processed_data:
    df = st.session_state.df
    ranking_df = st.session_state.ranking_df
    roadmap_df = st.session_state.roadmap_df
    
    # Generate Exec Report
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Executive Tools")
    
    report_md = f"""# AI Product Discovery Executive Summary
**Total Processed:** {len(df)} | **Duplicates Saved:** {st.session_state.total_duplicates} | **Unique Themes:** {len(ranking_df)}

## 🔥 NOW Priorities (High Friction & Impact)
"""
    now_items = roadmap_df[roadmap_df['Priority'] == '🔥 NOW']
    for _, item in now_items.iterrows():
        report_md += f"- **{item['Theme']}** (Score: {item['Score']:.1f}, Team: {item['Assigned Team']}): {item['Summary']}\n"
        
    report_md += "\n## ⭐ NEXT Priorities (Growth & Expansion)\n"
    next_items = roadmap_df[roadmap_df['Priority'] == '⭐ NEXT']
    for _, item in next_items.iterrows():
        report_md += f"- **{item['Theme']}** (Score: {item['Score']:.1f}, Team: {item['Assigned Team']}): {item['Summary']}\n"
        
    st.sidebar.download_button(label="📄 Download Exec Report (Markdown)", data=report_md, file_name='executive_summary.md', mime='text/markdown')

    # Export Data Buttons
    st.sidebar.subheader("📥 Export Reports")
    export_csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(label="📄 Export Full CSV", data=export_csv, file_name='product_discovery_full.csv', mime='text/csv')
    
    # Create UI Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Product Dashboard", "🧭 Impact/Effort Matrix", "🔍 Cluster Explorer", "💬 AI Search", "📈 Sentiment Trends"])

    with tab1:
        # High-Level Metrics (KPI Cards)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df)}</div><div class="kpi-label">Feedback Processed</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{st.session_state.total_duplicates}</div><div class="kpi-label">Duplicates Merged</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(ranking_df)}</div><div class="kpi-label">Unique Themes</div></div>', unsafe_allow_html=True)
        with col4:
            neg_count = len(df[df['Sentiment'] == 'NEGATIVE'])
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #ef4444;">{neg_count}</div><div class="kpi-label">High Friction Points</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("🗺️ AI Roadmap & Smart Routing")
        
        st.dataframe(
            roadmap_df.drop(columns=['ID']),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Priority Score",
                    help="Calculated priority",
                    format="%.1f",
                    min_value=0,
                    max_value=roadmap_df['Score'].max(),
                ),
            },
            hide_index=True,
            use_container_width=True
        )

    with tab2:
        st.header("🧭 Impact vs. Effort Matrix")
        st.write("Plot features visually to identify **Quick Wins** and **Time Sinks**. Use the sliders to estimate engineering effort (1-10).")
        
        # Let user adjust effort sliders for top 10
        col_sliders, col_plot = st.columns([1, 2])
        
        with col_sliders:
            st.write("### Adjust Effort")
            for i in range(min(10, len(roadmap_df))):
                theme = roadmap_df.iloc[i]['Theme']
                current_effort = roadmap_df.iloc[i]['Effort']
                new_e = st.slider(theme, 1, 10, int(current_effort), key=f"eff_{i}")
                roadmap_df.at[roadmap_df.index[i], 'Effort'] = new_e
                
        with col_plot:
            fig_matrix = plot_impact_effort_matrix(roadmap_df.head(10))
            st.pyplot(fig_matrix)
            
    with tab3:
        st.header("🔍 Deep Dive: Cluster Explorer")
        st.write("Investigate specific feature themes identified by the AI.")
        
        c_options = {int(row['ID']): row['Theme'] for _, row in roadmap_df.iterrows()}
        selected_c_id = st.selectbox("Select a Feature Theme to explore:", options=list(c_options.keys()), format_func=lambda x: c_options[x])
        
        if selected_c_id is not None:
            c_df = df[df['Cluster'] == selected_c_id]
            st.markdown("### 🤖 Extractive AI Summary")
            st.info(st.session_state.cluster_summaries[selected_c_id])
            
            st.markdown("### 🔑 Extracted Keywords")
            st.markdown(" ".join([f"`{k}`" for k in st.session_state.cluster_keywords[selected_c_id]]))
            
            st.markdown("### 📝 Raw Feature Requests within this Theme")
            st.dataframe(
                c_df[['Request_ID', 'Date', 'Sentiment', 'Auto_Tags', 'Request_Text', 'Upvotes']],
                hide_index=True,
                use_container_width=True
            )

    with tab4:
        st.header("💬 AI Product Search")
        st.write("Semantically search through thousands of requests instantly.")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., What are users saying about dark mode?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                embed_model = load_embedding_model()
                query_emb = embed_model.encode([prompt])[0]
                distances, indices = semantic_search(query_emb, st.session_state.index, k=5)
                
                response = f"I found the top {len(indices)} most relevant feature requests for **'{prompt}'**:\n\n"
                for i, idx in enumerate(indices):
                    if idx < len(df):
                        matched_row = df.iloc[idx]
                        response += f"> \"{matched_row['Request_Text']}\" *(Sentiment: {matched_row['Sentiment']}, Team: {matched_row['Auto_Tags']})*\n\n"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with tab5:
        st.header("📈 Sentiment & Distributions")
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.markdown("#### Feature Theme Distribution")
            fig_dist = plot_cluster_distribution(ranking_df)
            st.pyplot(fig_dist)
            
        with v_col2:
            st.markdown("#### Priority Share (Top 10)")
            fig_pie = plot_popularity_pie(ranking_df)
            st.pyplot(fig_pie)
            
        st.markdown("---")
        st.markdown("#### ⏳ Sentiment Trends Over Time")
        fig_trend = plot_sentiment_trends(df)
        st.pyplot(fig_trend)
        
        st.markdown("---")
        st.markdown("#### ☁️ Global Keyword Cloud")
        all_text = " ".join(df['Cleaned_Text'].tolist())
        fig_wc = generate_wordcloud(all_text)
        st.pyplot(fig_wc)

else:
    st.info("👈 Upload your customer feedback CSV or load the sample data to generate your AI Product Strategy dashboard.")
