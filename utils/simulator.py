import pandas as pd
import numpy as np
import plotly.graph_objects as go

SCENARIOS = {
    "Balanced (Standard RICE)": {
        "weight_reach": 1.0,
        "weight_impact": 1.0,
        "weight_confidence": 1.0,
        "weight_effort": 1.0,
        "enterprise_boost": 1.0,
        "description": "Standard product framework balancing reach, impact, confidence, and effort equally."
    },
    "Customer First (Churn Prevention)": {
        "weight_reach": 1.1,
        "weight_impact": 2.4,
        "weight_confidence": 1.2,
        "weight_effort": 0.8,
        "enterprise_boost": 1.2,
        "description": "Aggressively weights customer friction, negative sentiment, and pain points to reduce churn."
    },
    "Revenue First (Enterprise Growth)": {
        "weight_reach": 1.8,
        "weight_impact": 2.0,
        "weight_confidence": 1.1,
        "weight_effort": 0.9,
        "enterprise_boost": 3.0,
        "description": "Heavily weights Enterprise tier demand and high ACV contracts to drive new ARR."
    },
    "Quick Wins (Speed to Value)": {
        "weight_reach": 0.9,
        "weight_impact": 1.5,
        "weight_confidence": 1.0,
        "weight_effort": 2.5,
        "enterprise_boost": 0.9,
        "description": "Heavily penalizes development effort to accelerate shipping low-hanging fruit with high ROI."
    },
    "High Confidence & Security Focus": {
        "weight_reach": 1.0,
        "weight_impact": 1.8,
        "weight_confidence": 2.5,
        "weight_effort": 1.0,
        "enterprise_boost": 1.5,
        "description": "Favors highly validated, verified requests and foundational compliance over speculative bets."
    }
}

def calculate_scenario_score(
    reach: float,
    impact: float,
    confidence: float,
    effort: float,
    ent_count: int,
    total_requests: int,
    w_reach: float = 1.0,
    w_impact: float = 1.0,
    w_confidence: float = 1.0,
    w_effort: float = 1.0,
    enterprise_boost: float = 1.0
) -> float:
    """Calculates scenario RICE score with enterprise revenue multiplier."""
    safe_effort = max(0.5, float(effort) * max(0.1, float(w_effort)))
    eff_reach = float(reach) * max(0.1, float(w_reach))
    eff_impact = float(impact) * max(0.1, float(w_impact))
    eff_conf = float(confidence) * max(0.1, float(w_confidence))
    
    base_score = (eff_reach * eff_impact * eff_conf) / safe_effort
    
    # Enterprise boost multiplier
    if enterprise_boost != 1.0 and total_requests > 0:
        ent_ratio = float(ent_count) / float(total_requests)
        mult = 1.0 + (ent_ratio * (float(enterprise_boost) - 1.0))
        base_score *= mult
        
    return round(base_score, 1)

def simulate_ranking(ranking_df, weights, roadmap_df=None):
    """
    Computes simulated scores and ranks for a given set of scenario weights.
    """
    sim_df = ranking_df.copy()
    w_r = weights.get('weight_reach', 1.0)
    w_i = weights.get('weight_impact', 1.0)
    w_c = weights.get('weight_confidence', 1.0)
    w_e = weights.get('weight_effort', 1.0)
    ent_b = weights.get('enterprise_boost', 1.0)
    
    total_reqs = sim_df['Request Count'].sum() if 'Request Count' in sim_df.columns else 1000
    
    scores = []
    for _, row in sim_df.iterrows():
        reach = row.get('Reach', 100)
        impact = row.get('Impact', 1.0)
        conf = row.get('Confidence', 0.8)
        effort = row.get('Effort', 5.0)
        ent = row.get('Enterprise Count', 0)
        reqs = row.get('Request Count', 50)
        
        s = calculate_scenario_score(
            reach, impact, conf, effort, ent, reqs,
            w_reach=w_r, w_impact=w_i, w_confidence=w_c, w_effort=w_e,
            enterprise_boost=ent_b
        )
        scores.append(s)
        
    sim_df['Simulated_Score'] = scores
    sim_df['Simulated_Rank'] = sim_df['Simulated_Score'].rank(ascending=False, method='min').astype(int)
    
    # Attach themes if roadmap_df is available
    if roadmap_df is not None and 'Theme' in roadmap_df.columns and 'ID' in roadmap_df.columns:
        id_to_theme = dict(zip(roadmap_df['ID'], roadmap_df['Theme']))
        sim_df['Theme'] = sim_df['Cluster'].map(id_to_theme).fillna(sim_df['Cluster'].apply(lambda x: f"Cluster {x}"))
    elif 'Dominant Category' in sim_df.columns:
        sim_df['Theme'] = sim_df['Dominant Category']
    else:
        sim_df['Theme'] = sim_df['Cluster'].apply(lambda x: f"Feature Cluster {x}")
        
    return sim_df.sort_values(by='Simulated_Score', ascending=False).reset_index(drop=True)

def compare_two_scenarios(
    ranking_df,
    weights_a,
    weights_b,
    name_a="Customer First",
    name_b="Revenue First",
    roadmap_df=None
):
    """
    Simulates two scenarios side-by-side and calculates rank movement deltas.
    """
    sim_a = simulate_ranking(ranking_df, weights_a, roadmap_df)
    sim_b = simulate_ranking(ranking_df, weights_b, roadmap_df)
    
    # Merge on Cluster
    merged = pd.merge(
        sim_a[['Cluster', 'Theme', 'Simulated_Score', 'Simulated_Rank', 'Effort', 'Request Count', 'Negative Ratio', 'Enterprise Count']],
        sim_b[['Cluster', 'Simulated_Score', 'Simulated_Rank']],
        on='Cluster',
        suffixes=('_A', '_B')
    )
    
    merged['Rank_Delta'] = merged['Simulated_Rank_A'] - merged['Simulated_Rank_B']  # Positive = rose in Scenario B
    
    def format_movement(delta):
        if delta > 0:
            return f"▲ +{delta}"
        elif delta < 0:
            return f"▼ {delta}"
        else:
            return "— 0"
            
    merged['Movement'] = merged['Rank_Delta'].apply(format_movement)
    
    # Normalize display scores
    max_a = merged['Simulated_Score_A'].max() if merged['Simulated_Score_A'].max() > 0 else 1
    max_b = merged['Simulated_Score_B'].max() if merged['Simulated_Score_B'].max() > 0 else 1
    
    merged['Score_A_Norm'] = (merged['Simulated_Score_A'] / max_a * 10).round(1)
    merged['Score_B_Norm'] = (merged['Simulated_Score_B'] / max_b * 10).round(1)
    
    merged = merged.sort_values(by='Simulated_Rank_B').reset_index(drop=True)
    
    # Key insights
    top_gainers = merged.sort_values(by='Rank_Delta', ascending=False).head(3)
    top_decliners = merged.sort_values(by='Rank_Delta', ascending=True).head(3)
    
    return {
        "comparison_df": merged,
        "name_a": name_a,
        "name_b": name_b,
        "top_gainers": top_gainers,
        "top_decliners": top_decliners
    }

def plot_scenario_comparison_chart(comparison_df, name_a="Scenario A", name_b="Scenario B"):
    """
    Renders an elegant horizontal grouped bar chart comparing normalized scores across scenarios.
    """
    top_items = comparison_df.head(10).iloc[::-1]  # Reverse for top-down display
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_items['Theme'],
        x=top_items['Score_A_Norm'],
        name=name_a,
        orientation='h',
        marker=dict(color='#7C3AED', opacity=0.85, line=dict(color='#5B21B6', width=1)),
        hovertemplate='<b>%{y}</b><br>' + f'{name_a} Score: ' + '%{x:.1f}/10<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=top_items['Theme'],
        x=top_items['Score_B_Norm'],
        name=name_b,
        orientation='h',
        marker=dict(color='#10B981', opacity=0.85, line=dict(color='#047857', width=1)),
        hovertemplate='<b>%{y}</b><br>' + f'{name_b} Score: ' + '%{x:.1f}/10<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>Priority Score Comparison:</b> {name_a} vs {name_b}",
            font=dict(size=13, color="#0F172A", family="Plus Jakarta Sans, sans-serif"),
            x=0.0,
            y=0.98,
            xanchor="left",
            yanchor="top"
        ),
        barmode='group',
        bargap=0.25,
        bargroupgap=0.1,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=160, r=30, t=55, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#475569", family="Plus Jakarta Sans, sans-serif")
        ),
        xaxis=dict(
            title="Normalized Score (0 - 10)",
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            range=[0, 10.5]
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#334155", family="Plus Jakarta Sans, sans-serif")
        )
    )
    return fig
