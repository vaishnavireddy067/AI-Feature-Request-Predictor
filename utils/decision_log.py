import pandas as pd
from datetime import datetime

def init_decision_log(roadmap_df: pd.DataFrame) -> list:
    """
    Pre-populates an initial realistic decision log based on current roadmap rankings.
    """
    records = []
    if roadmap_df is None or roadmap_df.empty:
        return records
        
    for idx, row in roadmap_df.iterrows():
        theme = str(row['Theme'])
        score = float(row.get('Score', 8.0))
        stage = str(row.get('Stage', 'NOW')).upper()
        volume = int(row.get('Volume', 50))
        team = str(row.get('Assigned Team', 'Core Platform'))
        
        if stage == 'NOW':
            status = "Approved (Prioritized)"
            rationale = f"High RICE score ({score:.1f}) and strong customer demand ({volume} requests). Immediate sprint inclusion to resolve critical user pain points."
        elif stage == 'NEXT':
            status = "Planned (Target Q2)"
            rationale = f"Validated user demand ({volume} requests). Scheduled for upcoming quarter after foundational prerequisites are completed."
        else:
            status = "Deferred (Future Horizon)"
            rationale = f"Moderate impact relative to effort. Placed in backlog for reassessment in H2 roadmap cycle."
            
        records.append({
            "Decision_ID": f"DEC-{idx+1:03d}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Theme": theme,
            "Status": status,
            "Decision_Maker": "Vaishnavi Reddy",
            "Score_At_Decision": score,
            "Horizon": stage,
            "Category": team,
            "Customer_Requests": volume,
            "Rationale": rationale
        })
    return records

def record_new_decision(
    history: list,
    theme: str,
    status: str,
    rationale: str,
    decision_maker: str = "Vaishnavi Reddy",
    score: float = 8.0,
    horizon: str = "NOW",
    category: str = "Core Platform",
    requests: int = 50
) -> list:
    """Appends a new decision record with audit timestamp."""
    new_record = {
        "Decision_ID": f"DEC-{len(history)+1:03d}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Theme": theme,
        "Status": status,
        "Decision_Maker": decision_maker,
        "Score_At_Decision": float(score),
        "Horizon": horizon,
        "Category": category,
        "Customer_Requests": int(requests),
        "Rationale": rationale
    }
    history.insert(0, new_record)  # Newest first
    return history

def explain_decision_comparison(
    theme_a: str,
    theme_b: str,
    roadmap_df: pd.DataFrame,
    ranking_df: pd.DataFrame = None
) -> str:
    """
    Generates a structured, objective PM explanation comparing why Feature A was prioritized over Feature B.
    """
    row_a = roadmap_df[roadmap_df['Theme'] == theme_a]
    row_b = roadmap_df[roadmap_df['Theme'] == theme_b]
    
    if row_a.empty or row_b.empty:
        return "Please select two valid features to compare decision rationales."
        
    a = row_a.iloc[0]
    b = row_b.iloc[0]
    
    score_a = float(a.get('Score', 8.0))
    score_b = float(b.get('Score', 6.0))
    eff_a = float(a.get('Effort', 5.0))
    eff_b = float(b.get('Effort', 5.0))
    vol_a = int(a.get('Volume', 50))
    vol_b = int(b.get('Volume', 30))
    stage_a = str(a.get('Stage', 'NOW'))
    stage_b = str(b.get('Stage', 'NEXT'))
    
    score_delta = round(score_a - score_b, 1)
    vol_delta = vol_a - vol_b
    
    explanation = f"""### ⚖️ Strategic Decision Rationale: **{theme_a}** vs **{theme_b}**

| Comparison Factor | **{theme_a}** | **{theme_b}** | Delta / Trade-off |
| :--- | :--- | :--- | :--- |
| **Current Horizon** | `{stage_a}` | `{stage_b}` | {theme_a} scheduled earlier |
| **Smart RICE Score** | **{score_a:.1f}** | **{score_b:.1f}** | **{'+' if score_delta > 0 else ''}{score_delta:.1f} pts** ({theme_a} higher ROI) |
| **Customer Demand** | **{vol_a} requests** | **{vol_b} requests** | **{'+' if vol_delta > 0 else ''}{vol_delta} requests** ({(vol_a/vol_b - 1)*100:+.0f}% higher signal) |
| **Engineering Effort** | **{eff_a:.1f} pts** | **{eff_b:.1f} pts** | {f'{eff_a - eff_b:+.1f} pts effort difference'} |

#### 🎯 Executive Defense Summary:
1. **Demand & Reach Leverage:** **{theme_a}** addresses **{vol_a} customer requests**, outstripping demand for **{theme_b}** by {abs(vol_delta)} signals.
2. **Return on Engineering Investment (ROI):** With a RICE score of **{score_a:.1f}**, {theme_a} delivers greater business impact per unit of engineering capacity.
3. **Execution Horizon Recommendation:** **{theme_a}** is defensibly prioritized for **{stage_a}**, while **{theme_b}** is appropriately staged for **{stage_b}** pending foundational delivery.
"""
    return explanation

def export_decision_log_markdown(decision_history: list) -> str:
    """Exports decision log to Markdown audit trail."""
    md = "# 📜 Product Decision Log & Audit Trail\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Total Records:** {len(decision_history)}\n\n"
    md += "| ID | Timestamp | Feature Theme | Status | Horizon | RICE | Requests | Decided By | Rationale |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for r in decision_history:
        md += f"| {r['Decision_ID']} | {r['Timestamp']} | **{r['Theme']}** | `{r['Status']}` | {r['Horizon']} | {r['Score_At_Decision']:.1f} | {r['Customer_Requests']} | {r['Decision_Maker']} | {r['Rationale']} |\n"
        
    return md
