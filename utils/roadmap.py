import pandas as pd

def generate_ai_roadmap(df, ranking_df, cluster_keywords, cluster_summaries):
    """
    Generates a structured Now / Next / Later product roadmap.
    Uses RICE priority score, engineering effort, customer friction, and enterprise demand
    to intelligently partition features into actionable roadmap phases.
    """
    roadmap_items = []
    max_rice = ranking_df['RICE Score'].max() if not ranking_df.empty else 1.0
    median_rice = ranking_df['RICE Score'].median() if not ranking_df.empty else 0.5
    
    for _, row in ranking_df.iterrows():
        c_id = int(row['Cluster'])
        c_df = df[df['Cluster'] == c_id]
        
        # Theme Name
        if cluster_keywords.get(c_id):
            theme = cluster_keywords[c_id][0].title()
        else:
            theme = f"Theme {c_id}"
            
        score = float(row['RICE Score'])
        effort = float(row['Effort'])
        volume = int(row['Request Count'])
        upvotes = int(row['Total Upvotes'])
        neg_ratio = float(row.get('Negative Ratio', 0.0))
        
        # Team Tag
        all_tags = []
        if 'Auto_Tags' in c_df.columns:
            for tags in c_df['Auto_Tags']:
                if isinstance(tags, list):
                    all_tags.extend(tags)
                elif isinstance(tags, str):
                    all_tags.append(tags)
        assigned_team = pd.Series(all_tags).mode()[0] if all_tags else "General"
        
        # Priority Decision Logic (Now / Next / Later)
        if (score >= max_rice * 0.55) or (neg_ratio >= 35.0 and volume >= 15) or (effort <= 3.5 and score >= median_rice):
            stage = "NOW"
            stage_badge = "🔥 NOW (Current Sprint / Q1)"
            target_timeframe = "1 - 3 Months"
        elif score >= median_rice * 0.7 or volume >= 10:
            stage = "NEXT"
            stage_badge = "⭐ NEXT (Upcoming Quarter / Q2)"
            target_timeframe = "3 - 6 Months"
        else:
            stage = "LATER"
            stage_badge = "🧊 LATER (Future Horizon / Backlog)"
            target_timeframe = "6 - 12 Months"
            
        summary = cluster_summaries.get(c_id, "")
        if summary.startswith("Representative Request:"):
            clean_summary = summary.replace("Representative Request:", "").strip(' "')
        else:
            clean_summary = summary
            
        roadmap_items.append({
            "ID": c_id,
            "Theme": theme,
            "Stage": stage,
            "Priority": stage_badge,
            "Timeframe": target_timeframe,
            "RICE_Score": score,
            "Score": score,  # alias for compatibility
            "Effort": effort,
            "Volume": volume,
            "Upvotes": upvotes,
            "Assigned Team": assigned_team,
            "Negative_Sentiment_Pct": neg_ratio,
            "Summary": clean_summary
        })
        
    roadmap_df = pd.DataFrame(roadmap_items)
    stage_order = {"NOW": 0, "NEXT": 1, "LATER": 2}
    roadmap_df['stage_rank'] = roadmap_df['Stage'].map(stage_order)
    roadmap_df = roadmap_df.sort_values(by=['stage_rank', 'RICE_Score'], ascending=[True, False]).drop(columns=['stage_rank']).reset_index(drop=True)
    return roadmap_df

def move_roadmap_item(roadmap_df, theme_or_id, new_stage):
    """
    Moves an item in the roadmap to a new execution horizon (NOW, NEXT, or LATER)
    and updates timeframes and badges accordingly.
    """
    df_copy = roadmap_df.copy()
    stage_badges = {
        "NOW": ("🔥 NOW (Current Sprint / Q1)", "1 - 3 Months"),
        "NEXT": ("⭐ NEXT (Upcoming Quarter / Q2)", "3 - 6 Months"),
        "LATER": ("🧊 LATER (Future Horizon / Backlog)", "6 - 12 Months")
    }
    
    badge, timeframe = stage_badges.get(new_stage.upper(), ("PLAN", "TBD"))
    
    mask = (df_copy['Theme'] == theme_or_id) | (df_copy['ID'] == theme_or_id)
    if mask.any():
        df_copy.loc[mask, 'Stage'] = new_stage.upper()
        df_copy.loc[mask, 'Priority'] = badge
        df_copy.loc[mask, 'Timeframe'] = timeframe
        
    stage_order = {"NOW": 0, "NEXT": 1, "LATER": 2}
    df_copy['stage_rank'] = df_copy['Stage'].map(stage_order)
    df_copy = df_copy.sort_values(by=['stage_rank', 'RICE_Score'], ascending=[True, False]).drop(columns=['stage_rank']).reset_index(drop=True)
    return df_copy

def export_roadmap_markdown(roadmap_df):
    """Exports the roadmap to structured Markdown."""
    md = "# 🗺️ Product Roadmap (Now / Next / Later)\n\n"
    for stage, title in [("NOW", "🔥 NOW (Current Sprint & Q1)"), ("NEXT", "⭐ NEXT (Upcoming Quarter Q2)"), ("LATER", "🧊 LATER (Future Horizon)")]:
        items = roadmap_df[roadmap_df['Stage'] == stage]
        md += f"## {title}\n"
        if items.empty:
            md += "*No items scheduled in this horizon.*\n\n"
        else:
            for _, r in items.iterrows():
                md += f"- **{r['Theme']}** (RICE: {r['Score']} | Effort: {r['Effort']} pts | Team: {r['Assigned Team']})\n"
                md += f"  - *Summary:* {r['Summary']}\n"
            md += "\n"
    return md
