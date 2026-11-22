import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_demand_trends(df, ranking_df, cluster_keywords):
    """
    Analyzes historical velocity, demand momentum, and forecasts future trajectory
    for each feature theme based on its submission dates.
    """
    if 'Date' not in df.columns or df['Date'].dropna().empty:
        return pd.DataFrame(), {}
        
    df_copy = df.copy()
    df_copy['Datetime'] = pd.to_datetime(df_copy['Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Datetime'])
    
    if df_copy.empty:
        return pd.DataFrame(), {}
        
    max_date = df_copy['Datetime'].max()
    # Define recent window (last 60 days) vs prior window (60-120 days ago)
    recent_cutoff = max_date - timedelta(days=60)
    prior_cutoff = max_date - timedelta(days=120)
    
    theme_trends = []
    
    # Monthly aggregation overall
    df_copy['YearMonth'] = df_copy['Datetime'].dt.to_period('M')
    monthly_volume = df_copy.groupby('YearMonth').size()
    
    for _, row in ranking_df.iterrows():
        c_id = int(row['Cluster'])
        c_df = df_copy[df_copy['Cluster'] == c_id]
        total_vol = len(c_df)
        
        # Recent vs Prior window
        recent_count = len(c_df[c_df['Datetime'] >= recent_cutoff])
        prior_count = len(c_df[(c_df['Datetime'] < recent_cutoff) & (c_df['Datetime'] >= prior_cutoff)])
        
        # Momentum calculation: % change in velocity
        if prior_count > 0:
            growth_pct = ((recent_count - prior_count) / prior_count) * 100.0
        elif recent_count > 0:
            growth_pct = 100.0
        else:
            growth_pct = 0.0
            
        growth_pct = round(growth_pct, 1)
        
        # Momentum categorization
        if growth_pct >= 50.0 and recent_count >= 5:
            momentum = "🚀 Surging (High Velocity)"
            momentum_score = 3
        elif growth_pct >= 15.0:
            momentum = "📈 Gaining Traction"
            momentum_score = 2
        elif growth_pct >= -15.0:
            momentum = "⚖️ Steady Demand"
            momentum_score = 1
        else:
            momentum = "💤 Cooling Down"
            momentum_score = 0
            
        theme_name = cluster_keywords.get(c_id, ["Theme"])[0].title() if cluster_keywords.get(c_id) else f"Cluster {c_id}"
        
        # 30-Day Forward Forecast Projection
        daily_rate_recent = recent_count / 60.0
        projected_next_30d = int(round(daily_rate_recent * 30))
        
        theme_trends.append({
            "Cluster_ID": c_id,
            "Theme": theme_name,
            "Total_Requests": total_vol,
            "Recent_60d_Requests": recent_count,
            "Prior_60d_Requests": prior_count,
            "Growth_Rate_Pct": growth_pct,
            "Momentum": momentum,
            "Momentum_Score": momentum_score,
            "Forecast_30d_Requests": projected_next_30d,
            "RICE_Score": row['RICE Score']
        })
        
    trends_df = pd.DataFrame(theme_trends)
    if not trends_df.empty:
        trends_df = trends_df.sort_values(by=['Momentum_Score', 'Growth_Rate_Pct'], ascending=[False, False]).reset_index(drop=True)
        
    return trends_df, monthly_volume
