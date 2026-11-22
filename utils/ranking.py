import pandas as pd
import numpy as np

# Typical engineering complexity estimate per functional domain (scale 1 to 10)
DOMAIN_EFFORT_ESTIMATES = {
    "Security": 8.0,
    "Authentication": 7.0,
    "API": 6.5,
    "Payments": 7.5,
    "Mobile App": 6.0,
    "Performance": 5.5,
    "Reports": 4.5,
    "Dashboard": 4.0,
    "Notifications": 3.5,
    "UI/UX": 3.0,
    "General": 5.0
}

def estimate_theme_effort(category_name: str) -> float:
    """Estimates baseline engineering effort (1-10) based on domain complexity."""
    for domain, effort in DOMAIN_EFFORT_ESTIMATES.items():
        if domain.lower() in str(category_name).lower():
            return effort
    return 5.0

def calculate_rice_score(reach: float, impact: float, confidence: float, effort: float) -> float:
    """
    Calculates RICE Score: (Reach * Impact * Confidence) / Effort.
    Effort is clamped to minimum 0.5 to prevent division by zero.
    """
    safe_effort = max(0.5, float(effort))
    score = (float(reach) * float(impact) * float(confidence)) / safe_effort
    return round(score, 1)

def rank_features(df, cluster_col='Cluster', upvotes_col='Upvotes'):
    """
    Calculates Smart Priority Scores using the RICE framework for each feature cluster.
    Provides complete transparent, explainable breakdowns of Reach, Impact, Confidence, and Effort.
    """
    ranking_data = []
    total_dataset_requests = len(df)
    
    for cluster_id, group in df.groupby(cluster_col):
        size = len(group)
        total_upvotes = int(group[upvotes_col].sum()) if upvotes_col in group.columns else size
        
        # 1. REACH: Estimated monthly affected users
        # Weighted by user volume, upvotes, and enterprise customer presence
        ent_count = len(group[group['Customer_Segment'] == 'Enterprise']) if 'Customer_Segment' in group.columns else 0
        pro_count = len(group[group['Customer_Segment'] == 'Pro']) if 'Customer_Segment' in group.columns else 0
        free_count = len(group[group['Customer_Segment'] == 'Free']) if 'Customer_Segment' in group.columns else 0
        
        reach = (size * 10) + (total_upvotes * 2) + (ent_count * 20) + (pro_count * 5)
        
        # 2. IMPACT: Business Impact factor (0.5 to 3.0)
        # Higher if many negative sentiments (severe user friction/pain point) or high enterprise demand
        neg_count = len(group[group['Sentiment'] == 'NEGATIVE']) if 'Sentiment' in group.columns else 0
        neg_ratio = neg_count / size if size > 0 else 0
        ent_ratio = ent_count / size if size > 0 else 0
        
        if neg_ratio > 0.4 and ent_ratio > 0.3:
            impact = 3.0  # Massive impact (blocking top paying customers)
            impact_desc = "Massive (Critical Enterprise Friction)"
        elif neg_ratio > 0.3 or ent_ratio > 0.35:
            impact = 2.0  # High impact
            impact_desc = "High (Significant User Friction / Revenue Impact)"
        elif size > (total_dataset_requests * 0.08):
            impact = 1.5  # High volume medium impact
            impact_desc = "Medium-High (Broad User Demand)"
        elif free_count / size > 0.6:
            impact = 0.5  # Low impact
            impact_desc = "Low (Primarily Free Tier Requests)"
        else:
            impact = 1.0  # Standard Medium impact
            impact_desc = "Medium (Standard Improvement)"
            
        # 3. CONFIDENCE: Statistical confidence (50% to 100%, stored as 0.50 to 1.0)
        # More requests = higher statistical confidence
        if size >= 40:
            confidence = 1.0
            conf_desc = "100% (Very High Feedback Volume)"
        elif size >= 20:
            confidence = 0.9
            conf_desc = "90% (Substantial Sample Size)"
        elif size >= 10:
            confidence = 0.8
            conf_desc = "80% (Moderate Evidence)"
        elif size >= 5:
            confidence = 0.65
            conf_desc = "65% (Early Signal)"
        else:
            confidence = 0.5
            conf_desc = "50% (Low Sample Size)"
            
        # 4. EFFORT: Development Effort (1 to 10 scale)
        dominant_category = group['Category'].mode()[0] if 'Category' in group.columns and not group['Category'].empty else "General"
        effort = estimate_theme_effort(dominant_category)
        
        # 5. RICE SCORE
        rice_score = calculate_rice_score(reach, impact, confidence, effort)
        
        # Explanation text
        explanation = (
            f"Reach: {reach:,} affected users | "
            f"Impact: {impact} ({impact_desc}) | "
            f"Confidence: {int(confidence*100)}% ({conf_desc}) | "
            f"Effort: {effort} pts ({dominant_category})"
        )
        
        ranking_data.append({
            'Cluster': cluster_id,
            'Request Count': size,
            'Total Upvotes': total_upvotes,
            'Reach': reach,
            'Impact': impact,
            'Impact_Desc': impact_desc,
            'Confidence': confidence,
            'Confidence_Desc': conf_desc,
            'Effort': effort,
            'RICE Score': rice_score,
            'Popularity Score': rice_score, # backwards compatibility
            'Dominant Category': dominant_category,
            'Negative Ratio': round(neg_ratio * 100, 1),
            'Enterprise Count': ent_count,
            'Explanation': explanation
        })
        
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values(by='RICE Score', ascending=False).reset_index(drop=True)
    return ranking_df
