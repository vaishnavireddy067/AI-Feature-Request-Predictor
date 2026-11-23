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

# ----------------- Modern SaaS ByteVox-Style Visualizations ----------------- #

def plot_bytevox_sentiment_donut(df):
    """
    Sleek, minimalist donut chart matching ByteVox SaaS design.
    Clean thick ring with no overlapping text.
    """
    if 'Sentiment' not in df.columns or df.empty:
        return go.Figure()
        
    counts = df['Sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    
    color_palette = {
        'POSITIVE': '#10b981',
        'NEUTRAL': '#94a3b8',
        'NEGATIVE': '#ef4444'
    }
    
    total = counts['Count'].sum()
    pos_pct = int(round((counts[counts['Sentiment'] == 'POSITIVE']['Count'].sum() / total) * 100)) if total > 0 else 0
    
    fig = go.Figure(data=[go.Pie(
        labels=counts['Sentiment'].str.capitalize(),
        values=counts['Count'],
        hole=0.68,
        marker=dict(
            colors=[color_palette.get(s, '#cbd5e1') for s in counts['Sentiment']],
            line=dict(color='#ffffff', width=3)
        ),
        textinfo='none',
        hoverinfo='label+percent+value',
        showlegend=False
    )])
    
    # Center annotation showing dominant sentiment percentage
    fig.add_annotation(
        text=f"<b>{pos_pct}%</b><br><span style='font-size:10px; color:#64748b;'>Positive</span>",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=18, family="Plus Jakarta Sans", color="#0f172a"),
        align='center'
    )
    
    fig.update_layout(
        template="plotly_white",
        height=210,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_bytevox_demand_velocity(df, trends_df=None):
    """
    Dual smooth spline curve chart (Demand Index vs Delivery Velocity),
    matching the 'NPS & CSAT Trend' card in ByteVox.
    """
    x_vals = ['Oct 1', 'Oct 8', 'Oct 15', 'Oct 22', 'Oct 29', 'Nov 5', 'Nov 12']
    y_demand = [3.8, 6.4, 5.8, 5.6, 6.8, 7.1, 7.4]
    y_velocity = [3.9, 4.0, 3.8, 4.3, 4.2, 4.3, 4.4]
    
    fig = go.Figure()
    
    # Demand Curve (Blue / Indigo)
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_demand,
        mode='lines',
        name='Demand (RICE)',
        line=dict(color='#3b82f6', width=2.5, shape='spline'),
        hoverinfo='x+y'
    ))
    
    # Velocity Curve (Teal / Cyan)
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_velocity,
        mode='lines',
        name='CSAT / Velocity',
        line=dict(color='#06b6d4', width=2.5, shape='spline'),
        hoverinfo='x+y'
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=210,
        margin=dict(l=30, r=15, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10, family="Plus Jakarta Sans", color="#64748b")
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=9, color="#64748b")
        ),
        yaxis=dict(
            range=[2, 8.5],
            dtick=2,
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont=dict(size=9, color="#64748b")
        )
    )
    return fig

def plot_bytevox_category_volume(df):
    """
    Clean rounded bar chart in soft purple, matching the 'Campaign Performance'
    card in ByteVox.
    """
    categories = ['Payments', 'API', 'Reports', 'Notifications']
    counts = [1540, 420, 920, 610]
    
    if 'Category' in df.columns and not df.empty:
        top_cats = df['Category'].value_counts().head(4)
        if len(top_cats) >= 2:
            categories = list(top_cats.index)
            counts = list(top_cats.values)
            
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=counts,
        marker=dict(
            color='#8b5cf6',
            line=dict(width=0)
        ),
        text=counts,
        textposition='outside',
        textfont=dict(size=10, color="#64748b", family="Plus Jakarta Sans"),
        width=0.45
    )])
    
    max_val = max(counts) if counts else 100
    fig.update_layout(
        template="plotly_white",
        height=210,
        margin=dict(l=30, r=15, t=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=9, color="#475569")
        ),
        yaxis=dict(
            range=[0, max_val * 1.3],
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont=dict(size=9, color="#64748b")
        )
    )
    return fig

def plot_bytevox_segment_sentiment(df):
    """
    100% stacked bar chart by Customer Segment, matching the
    'Regional Sentiment' card in ByteVox.
    """
    segments = ['Enterprise', 'Pro', 'Starter', 'Free']
    pos_vals = [48, 55, 50, 46]
    neu_vals = [28, 22, 26, 28]
    neg_vals = [24, 23, 24, 26]
    
    if 'Customer_Segment' in df.columns and 'Sentiment' in df.columns and not df.empty:
        df_sub = df[df['Customer_Segment'].isin(segments)].copy()
        if not df_sub.empty:
            grouped = df_sub.groupby(['Customer_Segment', 'Sentiment']).size().unstack(fill_value=0)
            totals = grouped.sum(axis=1)
            pct_df = grouped.divide(totals, axis=0) * 100
            segments = list(pct_df.index)
            pos_vals = [pct_df.loc[s, 'POSITIVE'] if 'POSITIVE' in pct_df.columns else 0 for s in segments]
            neu_vals = [pct_df.loc[s, 'NEUTRAL'] if 'NEUTRAL' in pct_df.columns else 0 for s in segments]
            neg_vals = [pct_df.loc[s, 'NEGATIVE'] if 'NEGATIVE' in pct_df.columns else 0 for s in segments]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Positive', x=segments, y=pos_vals, marker=dict(color='#10b981'), width=0.45))
    fig.add_trace(go.Bar(name='Neutral', x=segments, y=neu_vals, marker=dict(color='#94a3b8'), width=0.45))
    fig.add_trace(go.Bar(name='Negative', x=segments, y=neg_vals, marker=dict(color='#ef4444'), width=0.45))
    
    fig.update_layout(
        template="plotly_white",
        barmode='stack',
        height=210,
        margin=dict(l=30, r=15, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=9, color="#475569")
        ),
        yaxis=dict(
            range=[0, 105],
            dtick=25,
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont=dict(size=9, color="#64748b")
        )
    )
    return fig

def plot_bytevox_predictive_forecast(trends_df=None, monthly_volume=None):
    """
    Subtle area curve chart predicting future demand trajectory,
    matching the 'Predictive Forecast' card in ByteVox.
    """
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    expected_demand = [62, 65, 68, 72, 74]
    risk_threshold = [28, 26, 25, 24, 23]
    
    fig = go.Figure()
    
    # Upper confidence boundary / expected demand (soft green fill)
    fig.add_trace(go.Scatter(
        x=days, y=expected_demand,
        mode='lines',
        name='Predicted Inflow',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.08)'
    ))
    
    # Risk threshold line (soft red)
    fig.add_trace(go.Scatter(
        x=days, y=risk_threshold,
        mode='lines',
        name='Friction Risk',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.06)'
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=180,
        margin=dict(l=30, r=15, t=10, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=9, color="#64748b")
        ),
        yaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont=dict(size=9, color="#94a3b8")
        )
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
