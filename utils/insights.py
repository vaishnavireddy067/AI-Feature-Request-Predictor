import pandas as pd
import numpy as np

def generate_evidence_backed_insights(df, ranking_df, cluster_keywords, cluster_summaries):
    """
    Generates rich, evidence-backed AI insights for each feature cluster.
    Returns a list of structured insight dicts with:
    - Theme & Cluster ID
    - AI Rationale ("Why AI made this recommendation")
    - Evidence Metrics (Request count, % of total feedback, total upvotes, segment breakdown)
    - Sentiment Breakdown (Positive %, Negative %, Neutral %)
    - Real Customer Quotes (top verbatim quotes)
    - Key Keywords
    - Actionable Product Recommendation
    """
    insights = []
    total_requests = len(df)
    
    for _, row in ranking_df.iterrows():
        c_id = int(row['Cluster'])
        c_df = df[df['Cluster'] == c_id]
        size = len(c_df)
        pct_of_total = round((size / total_requests) * 100, 1) if total_requests > 0 else 0
        total_upvotes = int(row['Total Upvotes'])
        
        # Segment breakdown
        segment_counts = c_df['Customer_Segment'].value_counts() if 'Customer_Segment' in c_df.columns else {}
        seg_str_list = [f"{seg}: {count} ({round(count/size*100)}%)" for seg, count in segment_counts.items()]
        seg_summary = ", ".join(seg_str_list) if seg_str_list else "All Segments"
        
        # Sentiment breakdown
        sent_counts = c_df['Sentiment'].value_counts() if 'Sentiment' in c_df.columns else {}
        neg_count = sent_counts.get('NEGATIVE', 0)
        pos_count = sent_counts.get('POSITIVE', 0)
        neu_count = sent_counts.get('NEUTRAL', 0)
        neg_pct = round((neg_count / size) * 100, 1) if size > 0 else 0
        pos_pct = round((pos_count / size) * 100, 1) if size > 0 else 0
        
        # Real verbatim customer quotes (pick highest upvoted and unique texts)
        sample_quotes = []
        sorted_samples = c_df.sort_values(by='Upvotes', ascending=False)
        seen_texts = set()
        for _, sample_row in sorted_samples.iterrows():
            text = str(sample_row['Request_Text']).strip()
            if text not in seen_texts:
                seen_texts.add(text)
                sample_quotes.append({
                    "text": text,
                    "upvotes": sample_row.get('Upvotes', 1),
                    "segment": sample_row.get('Customer_Segment', 'User'),
                    "date": sample_row.get('Date', ''),
                    "source": sample_row.get('Source', 'Direct')
                })
            if len(sample_quotes) >= 3:
                break
                
        # Theme label
        theme_label = cluster_keywords.get(c_id, ["Feature"])[0].title() if cluster_keywords.get(c_id) else f"Cluster {c_id}"
        
        # AI Rationale & Actionable Recommendation formulation
        ent_count = segment_counts.get('Enterprise', 0)
        ent_pct = round((ent_count / size) * 100) if size > 0 else 0
        effort = row['Effort']
        rice_score = row['RICE Score']
        
        if neg_pct >= 40 and ent_pct >= 25:
            rationale_badge = "🚨 Critical Enterprise Blocker"
            rationale = (
                f"Severe customer frustration ({neg_pct}% negative sentiment) paired with high enterprise "
                f"concentration ({ent_pct}% of requests from Enterprise tier). This represents a direct churn risk."
            )
            recommendation = "Immediate Sprint Inclusion: Assign to senior tech lead to unblock enterprise renewal accounts."
            urgency = "High"
        elif effort <= 4.0 and rice_score >= (ranking_df['RICE Score'].median()):
            rationale_badge = "⚡ Fast ROI Quick Win"
            rationale = (
                f"High return on engineering investment: Low implementation effort ({effort} points) "
                f"with substantial demand across {size} customers and strong RICE score ({rice_score})."
            )
            recommendation = "Slot into upcoming sprint as a quick win to boost customer satisfaction score (CSAT)."
            urgency = "Medium-High"
        elif ent_pct >= 40:
            rationale_badge = "💼 Strategic Enterprise Upsell"
            rationale = (
                f"Heavily skewed toward Enterprise customers ({ent_pct}% of total requests). "
                f"Delivering this feature directly supports annual contract value (ACV) expansion and competitive parity."
            )
            recommendation = "Product Discovery & Specification: Align with sales and solutions engineering on compliance needs."
            urgency = "High"
        elif size > (total_requests * 0.1):
            rationale_badge = "👥 Broad Market Demand"
            rationale = (
                f"Affects a significant cross-section of your user base ({pct_of_total}% of all customer feedback). "
                f"Broad appeal will drive organic word-of-mouth adoption and engagement."
            )
            recommendation = "Plan for Next Release: Conduct UX benchmarking to ensure intuitive self-serve design."
            urgency = "Medium"
        else:
            rationale_badge = "💡 Incremental Polish"
            rationale = (
                f"Moderate interest across {size} requests with steady engagement. "
                f"Contributes to product maturity and overall usability."
            )
            recommendation = "Add to product backlog and bundle with related UX enhancements in the next quarter."
            urgency = "Low"
            
        insights.append({
            "Cluster_ID": c_id,
            "Theme": theme_label,
            "RICE_Score": rice_score,
            "Request_Count": size,
            "Pct_Of_Total": pct_of_total,
            "Total_Upvotes": total_upvotes,
            "Rationale_Badge": rationale_badge,
            "AI_Rationale": rationale,
            "Actionable_Recommendation": recommendation,
            "Urgency": urgency,
            "Segment_Summary": seg_summary,
            "Segment_Breakdown": segment_counts.to_dict(),
            "Sentiment_Breakdown": {
                "Negative_Pct": neg_pct,
                "Positive_Pct": pos_pct,
                "Neutral_Pct": round(100 - neg_pct - pos_pct, 1)
            },
            "Customer_Quotes": sample_quotes,
            "Keywords": cluster_keywords.get(c_id, []),
            "Summary": cluster_summaries.get(c_id, "")
        })
        
    return insights
