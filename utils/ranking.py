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

SCORING_PRESETS = {
    "Balanced (Standard RICE)": {
        "weight_reach": 1.0,
        "weight_impact": 1.0,
        "weight_confidence": 1.0,
        "weight_effort": 1.0,
        "description": "Standard product framework balancing reach, impact, confidence, and effort equally."
    },
    "Customer Friction & Churn First": {
        "weight_reach": 1.1,
        "weight_impact": 1.8,
        "weight_confidence": 1.0,
        "weight_effort": 0.8,
        "description": "Prioritizes high-impact friction points and negative sentiment to reduce churn."
    },
    "Quick Wins & High ROI": {
        "weight_reach": 0.9,
        "weight_impact": 1.4,
        "weight_confidence": 1.0,
        "weight_effort": 1.8,
        "description": "Heavily penalizes development effort to accelerate shipping high-ROI quick wins."
    },
    "Enterprise ARR Expansion": {
        "weight_reach": 1.7,
        "weight_impact": 1.6,
        "weight_confidence": 1.2,
        "weight_effort": 0.9,
        "description": "Emphasizes enterprise tier demand to maximize Annual Contract Value (ACV)."
    }
}

def estimate_theme_effort(category_name: str) -> float:
    """Estimates baseline engineering effort (1-10) based on domain complexity."""
    for domain, effort in DOMAIN_EFFORT_ESTIMATES.items():
        if domain.lower() in str(category_name).lower():
            return effort
    return 5.0

def calculate_rice_score(
    reach: float, 
    impact: float, 
    confidence: float, 
    effort: float,
    weight_reach: float = 1.0,
    weight_impact: float = 1.0,
    weight_confidence: float = 1.0,
    weight_effort: float = 1.0
) -> float:
    """
    Calculates Smart Priority Score using the Weighted RICE Framework:
    
    FORMULA:
        Smart RICE = ((Reach * W_R) * (Impact * W_I) * (Confidence * W_C)) / max(0.5, Effort * W_E)
        
    INPUT SCALES & DEFINITIONS:
        - Reach (50 to 1,500): Estimated monthly affected users based on:
          Reach = (Requests * 10) + (Upvotes * 2) + (Enterprise_Count * 20) + (Pro_Count * 5)
          W_R (0.2 - 2.5): Weight on broad user volume vs segment focus.
          
        - Impact (0.5 to 3.0 scale): Qualitative business and user retention impact.
          3.0 = Massive (Critical Enterprise blocker with >40% negative sentiment)
          2.0 = High (Significant user friction or major revenue driver)
          1.5 = Medium-High (Broad user demand across >8% of feedback)
          1.0 = Medium (Standard expected functional enhancement)
          0.5 = Low (Minor cosmetic polish, primarily free-tier)
          W_I (0.2 - 2.5): Weight on customer friction and churn prevention.
          
        - Confidence (0.50 to 1.0 scale): Statistical evidence confidence based on sample size.
          1.0  = 100% (Very High sample: >= 40 requests)
          0.9  = 90% (Substantial sample: >= 20 requests)
          0.8  = 80% (Moderate evidence: >= 10 requests)
          0.65 = 65% (Early signal: >= 5 requests)
          0.5  = 50% (Low sample: < 5 requests)
          W_C (0.2 - 2.5): Penalizes speculative or unverified requests.
          
        - Effort (1.0 to 10.0 scale): Domain engineering complexity in story points.
          UI/UX = 3.0, Notifications = 3.5, Dashboard = 4.0, Reports = 4.5,
          Performance = 5.5, Mobile = 6.0, API = 6.5, Payments = 7.5, Security = 8.0.
          Effort denominator is clamped to min 0.5 to prevent division by zero.
          W_E (0.2 - 2.5): Effort penalty weight. Higher W_E favors quick wins.
          
    BASELINE:
        When all weights = 1.0, this simplifies identically to standard RICE:
        (Reach * Impact * Confidence) / Effort.
    """
    safe_effort = max(0.5, float(effort) * max(0.1, float(weight_effort)))
    effective_reach = float(reach) * max(0.1, float(weight_reach))
    effective_impact = float(impact) * max(0.1, float(weight_impact))
    effective_conf = float(confidence) * max(0.1, float(weight_confidence))
    
    score = (effective_reach * effective_impact * effective_conf) / safe_effort
    return round(score, 1)

def rank_features(
    df, 
    cluster_col='Cluster', 
    upvotes_col='Upvotes',
    weight_reach: float = 1.0,
    weight_impact: float = 1.0,
    weight_confidence: float = 1.0,
    weight_effort: float = 1.0
):
    """
    Calculates Smart Priority Scores using the RICE framework for each feature cluster.
    Supports custom scoring weights for Reach, Impact, Confidence, and Effort.
    Provides complete transparent, explainable breakdowns of Reach, Impact, Confidence, and Effort.
    """
    ranking_data = []
    total_dataset_requests = len(df)
    
    for cluster_id, group in df.groupby(cluster_col):
        size = len(group)
        total_upvotes = int(group[upvotes_col].sum()) if upvotes_col in group.columns else size
        
        # 1. REACH: Estimated monthly affected users
        ent_count = len(group[group['Customer_Segment'] == 'Enterprise']) if 'Customer_Segment' in group.columns else 0
        pro_count = len(group[group['Customer_Segment'] == 'Pro']) if 'Customer_Segment' in group.columns else 0
        free_count = len(group[group['Customer_Segment'] == 'Free']) if 'Customer_Segment' in group.columns else 0
        
        reach = (size * 10) + (total_upvotes * 2) + (ent_count * 20) + (pro_count * 5)
        
        # 2. IMPACT: Business Impact factor (0.5 to 3.0)
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
        
        # 5. RICE SCORE (with weights)
        rice_score = calculate_rice_score(
            reach, impact, confidence, effort,
            weight_reach, weight_impact, weight_confidence, weight_effort
        )
        
        # Base RICE score (unweighted standard 1.0) for comparison
        base_rice_score = calculate_rice_score(reach, impact, confidence, effort)
        
        # Explanation text
        explanation = (
            f"Reach: {reach:,} (w={weight_reach}) | "
            f"Impact: {impact} (w={weight_impact}, {impact_desc}) | "
            f"Confidence: {int(confidence*100)}% (w={weight_confidence}) | "
            f"Effort: {effort} pts (w={weight_effort}, {dominant_category})"
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
            'Base RICE Score': base_rice_score,
            'Popularity Score': rice_score, # backwards compatibility
            'Dominant Category': dominant_category,
            'Negative Ratio': round(neg_ratio * 100, 1),
            'Enterprise Count': ent_count,
            'Explanation': explanation
        })
        
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values(by='RICE Score', ascending=False).reset_index(drop=True)
    return ranking_df
