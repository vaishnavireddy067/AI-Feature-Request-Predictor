import pandas as pd

def detect_emerging_feature_alerts(trends_df, ranking_df, df):
    """
    Detects critical early signals and emerging feature alerts:
    1. 🚀 Velocity Surge: Unprecedented growth rate in the last 60 days
    2. 🚨 Friction Spike: Severe customer negative sentiment and pain points
    3. 💼 Enterprise Breakout: Outsized concentration of Enterprise tier requests
    4. ⚡ Sleeper Feature: Low overall backlog rank but sudden aggressive acceleration
    """
    alerts = []
    
    if trends_df is None or trends_df.empty:
        return alerts
        
    for _, row in trends_df.iterrows():
        theme = str(row['Theme'])
        c_id = int(row['Cluster_ID'])
        growth_pct = float(row['Growth_Rate_Pct'])
        recent_count = int(row['Recent_60d_Requests'])
        prior_count = int(row['Prior_60d_Requests'])
        
        # Pull matching ranking row
        rank_row = ranking_df[ranking_df['Cluster'] == c_id]
        neg_ratio = float(rank_row['Negative Ratio'].values[0]) if not rank_row.empty else 0.0
        ent_count = int(rank_row['Enterprise Count'].values[0]) if not rank_row.empty else 0
        total_reqs = int(row['Total_Requests'])
        ent_ratio = (ent_count / total_reqs * 100) if total_reqs > 0 else 0
        
        # 1. 🚨 Severe Friction Spike
        if neg_ratio >= 35.0 and recent_count >= 6:
            alerts.append({
                "Theme": theme,
                "Type": "Friction Spike",
                "Badge": "🚨 CRITICAL FRICTION",
                "Severity": "Critical",
                "Metric": f"{neg_ratio}% Negative Sentiment ({recent_count} recent tickets)",
                "Description": (
                    f"Customers are actively expressing severe frustration regarding '{theme}'. "
                    f"This is a primary driver of support ticket volume and user churn."
                ),
                "Action": "Immediate Bug / UX Spike: Fast-track resolution into current sprint to prevent churn."
            })
            
        # 2. 🚀 Velocity Surge
        if growth_pct >= 45.0 and recent_count >= 8:
            alerts.append({
                "Theme": theme,
                "Type": "Velocity Surge",
                "Badge": "🚀 VELOCITY SURGE",
                "Severity": "High",
                "Metric": f"+{growth_pct}% 60-Day Growth ({recent_count} vs {prior_count} prev)",
                "Description": (
                    f"Customer demand for '{theme}' is skyrocketing at +{growth_pct}% acceleration. "
                    f"Early indicators show this will become your #1 requested feature next quarter."
                ),
                "Action": "Proactive PM Discovery: Formulate PRD and design specs before request volume overwhelms backlog."
            })
            
        # 3. 💼 Enterprise Breakout
        if ent_ratio >= 40.0 and ent_count >= 5:
            alerts.append({
                "Theme": theme,
                "Type": "Enterprise Breakout",
                "Badge": "💼 ENTERPRISE BREAKOUT",
                "Severity": "High",
                "Metric": f"{int(ent_ratio)}% Enterprise Tier Concentration ({ent_count} accounts)",
                "Description": (
                    f"High-revenue Enterprise accounts are disproportionately demanding '{theme}'. "
                    f"Unblocking this feature directly correlates with enterprise sales win-rate and retention."
                ),
                "Action": "Sales & Solutions Alignment: Sync with Enterprise Account Executives on contract SLA requirements."
            })
            
        # 4. ⚡ Sleeper Feature
        if total_reqs < 20 and growth_pct >= 60.0 and recent_count >= 5:
            alerts.append({
                "Theme": theme,
                "Type": "Sleeper Hit",
                "Badge": "⚡ SLEEPER FEATURE",
                "Severity": "Opportunity",
                "Metric": f"+{growth_pct}% sudden acceleration from low baseline",
                "Description": (
                    f"'{theme}' has modest overall volume but has spiked abruptly in the recent cycle. "
                    f"Represents a potential market differentiator or emergent user workflow."
                ),
                "Action": "Exploratory Validation: Conduct 3 user interviews to determine if this is a broader trend."
            })
            
    # Sort alerts by severity: Critical > High > Opportunity
    severity_order = {"Critical": 0, "High": 1, "Opportunity": 2}
    alerts.sort(key=lambda x: severity_order.get(x["Severity"], 3))
    return alerts
