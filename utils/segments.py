import pandas as pd

def analyze_customer_segments(df, ranking_df, cluster_keywords):
    """
    Performs comprehensive Customer Segment Analysis.
    Compares feature priorities, friction points, and demand across:
    - Enterprise (high ACV, compliance, security, SSO)
    - Pro (power users, integrations, analytics)
    - Starter (growing SMBs, billing, core workflows)
    - Free (self-serve, UX, mobile)
    """
    if 'Customer_Segment' not in df.columns:
        return {}, pd.DataFrame(), pd.DataFrame()
        
    # 1. Overall Segment Metrics
    segment_stats = []
    total_records = len(df)
    
    for seg, seg_group in df.groupby('Customer_Segment'):
        count = len(seg_group)
        pct = round((count / total_records) * 100, 1)
        upvotes = int(seg_group['Upvotes'].sum()) if 'Upvotes' in seg_group.columns else count
        neg_count = len(seg_group[seg_group['Sentiment'] == 'NEGATIVE']) if 'Sentiment' in seg_group.columns else 0
        neg_pct = round((neg_count / count) * 100, 1) if count > 0 else 0
        
        segment_stats.append({
            "Segment": seg,
            "Request_Count": count,
            "Share_Pct": pct,
            "Total_Upvotes": upvotes,
            "Negative_Sentiment_Pct": neg_pct,
            "Avg_Upvotes_Per_Req": round(upvotes / count, 1) if count > 0 else 1
        })
        
    segment_summary_df = pd.DataFrame(segment_stats).sort_values(by="Request_Count", ascending=False).reset_index(drop=True)
    
    # 2. Segment-to-Theme Cross-Tabulation
    # Map Cluster ID to human-readable Theme
    id_to_theme = {
        int(row['Cluster']): (cluster_keywords.get(int(row['Cluster']), ["Theme"])[0].title() if cluster_keywords.get(int(row['Cluster'])) else f"Theme {row['Cluster']}")
        for _, row in ranking_df.iterrows()
    }
    
    df_copy = df.copy()
    df_copy['Theme_Name'] = df_copy['Cluster'].map(id_to_theme).fillna("Other")
    
    cross_tab = pd.crosstab(df_copy['Theme_Name'], df_copy['Customer_Segment'])
    
    # Calculate Top Theme for each Segment
    top_per_segment = {}
    for col in cross_tab.columns:
        top_series = cross_tab[col].sort_values(ascending=False).head(5)
        top_per_segment[col] = [
            {"Theme": theme, "Count": count, "Share_In_Segment": round((count / cross_tab[col].sum()) * 100, 1)}
            for theme, count in top_series.items()
        ]
        
    # 3. Revenue-Weighted Scoring (Enterprise = 4x, Pro = 2x, Starter = 1.5x, Free = 1x)
    weights = {"Enterprise": 4.0, "Pro": 2.0, "Starter": 1.5, "Free": 1.0}
    weighted_scores = []
    
    for _, row in ranking_df.iterrows():
        c_id = int(row['Cluster'])
        c_df = df[df['Cluster'] == c_id]
        
        score_val = 0.0
        for seg, mult in weights.items():
            seg_len = len(c_df[c_df['Customer_Segment'] == seg]) if 'Customer_Segment' in c_df.columns else 0
            score_val += (seg_len * mult)
            
        theme_title = id_to_theme.get(c_id, f"Theme {c_id}")
        weighted_scores.append({
            "Cluster_ID": c_id,
            "Theme": theme_title,
            "RICE_Score": row['RICE Score'],
            "Revenue_Weighted_Score": round(score_val, 1),
            "Enterprise_Requests": len(c_df[c_df['Customer_Segment'] == 'Enterprise']) if 'Customer_Segment' in c_df.columns else 0,
            "Pro_Requests": len(c_df[c_df['Customer_Segment'] == 'Pro']) if 'Customer_Segment' in c_df.columns else 0,
            "Free_Requests": len(c_df[c_df['Customer_Segment'] == 'Free']) if 'Customer_Segment' in c_df.columns else 0
        })
        
    revenue_df = pd.DataFrame(weighted_scores).sort_values(by="Revenue_Weighted_Score", ascending=False).reset_index(drop=True)
    
    return top_per_segment, segment_summary_df, revenue_df
