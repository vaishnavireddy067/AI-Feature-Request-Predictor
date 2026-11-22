import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_interactive_rice_matrix(roadmap_df):
    """
    Generates an interactive Plotly Impact vs Effort bubble matrix.
    X-axis: Effort (1-10)
    Y-axis: Impact (0.5-3.0) or RICE Score
    Bubble Size: Volume or Reach
    Color: Roadmap Stage (Now, Next, Later)
    """
    df_plot = roadmap_df.copy()
    
    color_map = {
        "NOW": "#ef4444",
        "NEXT": "#f59e0b",
        "LATER": "#3b82f6"
    }
    
    fig = px.scatter(
        df_plot,
        x="Effort",
        y="Score",
        size="Volume",
        color="Stage" if "Stage" in df_plot.columns else None,
        color_discrete_map=color_map,
        hover_name="Theme",
        hover_data={
            "Score": True,
            "Effort": True,
            "Volume": True,
            "Assigned Team": True,
            "Summary": True
        },
        labels={
            "Effort": "Engineering Effort (1 = Low, 10 = High)",
            "Score": "RICE Priority Score",
            "Volume": "Feedback Volume"
        },
        title="<b>Interactive RICE Priority Matrix (Impact vs. Effort)</b>"
    )
    
    # Quadrant Dividers
    max_score = df_plot['Score'].max() if not df_plot.empty else 100
    mid_score = max_score / 2.0
    
    fig.add_shape(type="line", x0=5, y0=0, x1=5, y1=max_score*1.1, line=dict(color="rgba(156, 163, 175, 0.4)", dash="dash"))
    fig.add_shape(type="line", x0=0, y0=mid_score, x1=10.5, y1=mid_score, line=dict(color="rgba(156, 163, 175, 0.4)", dash="dash"))
    
    # Quadrant Annotations
    fig.add_annotation(x=2.5, y=max_score*0.95, text="⭐ <b>Quick Wins</b><br>(High Impact, Low Effort)", showarrow=False, font=dict(color="rgba(16, 185, 129, 0.8)", size=12))
    fig.add_annotation(x=7.5, y=max_score*0.95, text="🏗️ <b>Strategic Bets</b><br>(High Impact, High Effort)", showarrow=False, font=dict(color="rgba(59, 130, 246, 0.8)", size=12))
    fig.add_annotation(x=2.5, y=mid_score*0.2, text="🩹 <b>Fill-ins</b><br>(Low Impact, Low Effort)", showarrow=False, font=dict(color="rgba(156, 163, 175, 0.8)", size=12))
    fig.add_annotation(x=7.5, y=mid_score*0.2, text="🛑 <b>Time Sinks</b><br>(Low Impact, High Effort)", showarrow=False, font=dict(color="rgba(239, 68, 68, 0.8)", size=12))
    
    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(range=[0.5, 10.5], showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6")
    )
    return fig

def plot_interactive_trends(df, trends_df=None):
    """Generates an interactive monthly volume trend line with sentiment breakdown."""
    if 'Date' not in df.columns:
        return go.Figure()
        
    df_copy = df.copy()
    df_copy['Datetime'] = pd.to_datetime(df_copy['Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Datetime'])
    df_copy['Month'] = df_copy['Datetime'].dt.to_period('M').astype(str)
    
    monthly = df_copy.groupby(['Month', 'Sentiment']).size().reset_index(name='Count')
    
    color_map = {
        'POSITIVE': '#10b981',
        'NEGATIVE': '#ef4444',
        'NEUTRAL': '#94a3b8'
    }
    
    fig = px.bar(
        monthly, 
        x="Month", 
        y="Count", 
        color="Sentiment",
        color_discrete_map=color_map,
        title="<b>Monthly Customer Feedback Volume & Sentiment Trends</b>",
        labels={"Count": "Requests", "Month": "Month"}
    )
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6")
    )
    return fig

def plot_segment_donut(df):
    """Plots interactive donut chart of customer segment distribution."""
    if 'Customer_Segment' not in df.columns:
        return go.Figure()
        
    counts = df['Customer_Segment'].value_counts().reset_index()
    counts.columns = ['Segment', 'Count']
    
    colors = ['#4f46e5', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b']
    
    fig = px.pie(
        counts, 
        names='Segment', 
        values='Count', 
        hole=0.45,
        title="<b>Feedback Share by Customer Segment</b>",
        color_discrete_sequence=colors
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=2)))
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_cluster_distribution(ranking_df):
    """Plots bar chart of the number of requests per cluster."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    top_items = ranking_df.head(12)
    bars = ax.bar(top_items['Dominant Category'].astype(str), top_items['Request Count'], color='#4f46e5', width=0.6)
    ax.set_title('Top Feature Themes by Request Count', fontsize=13, fontweight='bold', color='#1f2937')
    ax.set_ylabel('Number of Requests', fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    return fig

def generate_wordcloud(text):
    """Generates a word cloud from the given text."""
    wordcloud = WordCloud(
        width=800, height=360, 
        background_color='white',
        colormap='magma',
        max_words=120
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig
