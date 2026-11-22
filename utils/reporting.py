import pandas as pd
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_markdown_report(df, ranking_df, roadmap_df, insights, trends_df=None, segment_df=None):
    """
    Generates a comprehensive executive Markdown report.
    """
    now_str = datetime.now().strftime("%B %d, %Y")
    total_feedback = len(df)
    unique_themes = len(ranking_df)
    
    neg_count = len(df[df['Sentiment'] == 'NEGATIVE']) if 'Sentiment' in df.columns else 0
    neg_pct = round(neg_count / total_feedback * 100, 1) if total_feedback > 0 else 0
    segments_list = df['Customer_Segment'].value_counts().index.tolist() if 'Customer_Segment' in df.columns else ['All Users']
    
    md = f"""# 🚀 Executive Product Strategy & Feedback Intelligence Report
**Generated on:** {now_str} | **Scope:** Enterprise Product Roadmap & Customer Voice Analysis

---

## 1. Executive Summary & Core KPIs

- **Total Feedback Processed:** {total_feedback:,} customer requests
- **Unique Strategic Themes Identified:** {unique_themes}
- **High Friction Feedback:** {neg_count} ({neg_pct}%)
- **Primary Customer Segments:** {', '.join(segments_list)}

---

## 2. Top Ranked Features (Smart RICE Framework)

The RICE scoring model (*Reach × Impact × Confidence / Effort*) prioritizes high-leverage features:

| Rank | Feature Theme | RICE Score | Reach | Impact | Conf. | Effort | Dominant Domain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for i, row in ranking_df.head(10).iterrows():
        md += f"| #{i+1} | **{row['Dominant Category']}** | **{row['RICE Score']}** | {row['Reach']:,} | {row['Impact']} | {int(row['Confidence']*100)}% | {row['Effort']} | {row['Dominant Category']} |\n"

    md += """
---

## 3. Now / Next / Later Strategic Roadmap

### 🔥 NOW (Current Sprint & Q1 Execution - High Friction / High ROI)
"""
    now_items = roadmap_df[roadmap_df['Stage'] == 'NOW'] if 'Stage' in roadmap_df.columns else roadmap_df.head(3)
    for _, item in now_items.iterrows():
        md += f"- **{item['Theme']}** (RICE: {item['Score']}, Effort: {item['Effort']} pts, Team: {item['Assigned Team']})\n"
        md += f"  - *Summary:* {item['Summary']}\n"

    md += "\n### ⭐ NEXT (Upcoming Quarter Q2 - Strategic Expansion)\n"
    next_items = roadmap_df[roadmap_df['Stage'] == 'NEXT'] if 'Stage' in roadmap_df.columns else roadmap_df.iloc[3:7]
    for _, item in next_items.iterrows():
        md += f"- **{item['Theme']}** (RICE: {item['Score']}, Effort: {item['Effort']} pts, Team: {item['Assigned Team']})\n"
        md += f"  - *Summary:* {item['Summary']}\n"

    md += "\n### 🧊 LATER (Future Horizon & Backlog - Exploratory / High Effort)\n"
    later_items = roadmap_df[roadmap_df['Stage'] == 'LATER'] if 'Stage' in roadmap_df.columns else roadmap_df.iloc[7:]
    for _, item in later_items.iterrows():
        md += f"- **{item['Theme']}** (RICE: {item['Score']}, Effort: {item['Effort']} pts, Team: {item['Assigned Team']})\n"
        md += f"  - *Summary:* {item['Summary']}\n"

    md += """
---

## 4. Evidence-Backed Customer Insights & Verbatim Quotes

"""
    for ins in insights[:5]:
        md += f"### {ins['Rationale_Badge']}: {ins['Theme']}\n"
        md += f"- **AI Rationale:** {ins['AI_Rationale']}\n"
        md += f"- **Evidence:** {ins['Request_Count']} requests ({ins['Pct_Of_Total']}% of total) | {ins['Total_Upvotes']} upvotes\n"
        md += f"- **Customer Segments:** {ins['Segment_Summary']}\n"
        md += f"- **Action Recommendation:** {ins['Actionable_Recommendation']}\n"
        md += "- **Representative Customer Quotes:**\n"
        for q in ins['Customer_Quotes']:
            md += f"  > \"{q['text']}\" — *{q['segment']} Tier via {q['source']}*\n"
        md += "\n"

    if trends_df is not None and not trends_df.empty:
        md += """---

## 5. Demand Trend Momentum & Forecasting

Features gaining velocity before reaching peak backlog volume:

| Feature Theme | Momentum Status | Growth Rate (60d) | Recent Volume | 30d Projected |
| :--- | :--- | :---: | :---: | :---: |
"""
        for _, t_row in trends_df.head(6).iterrows():
            md += f"| **{t_row['Theme']}** | {t_row['Momentum']} | +{t_row['Growth_Rate_Pct']}% | {t_row['Recent_60d_Requests']} | ~{t_row['Forecast_30d_Requests']} reqs |\n"

    md += """
---
*Confidential Executive Report — Powered by AI Feature Request Prioritizer*
"""
    return md

def generate_html_report(df, ranking_df, roadmap_df, insights, trends_df=None, segment_df=None):
    """Generates a standalone, beautifully styled HTML executive report."""
    now_str = datetime.now().strftime("%B %d, %Y")
    total_feedback = len(df)
    neg_count = len(df[df['Sentiment'] == 'NEGATIVE']) if 'Sentiment' in df.columns else 0
    top_rice = ranking_df['RICE Score'].max() if not ranking_df.empty else 0
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Product Strategy - Executive Summary</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px auto; max-width: 1000px; color: #1f2937; line-height: 1.6; background-color: #f9fafb; padding: 20px; }}
    .card {{ background: white; border-radius: 12px; padding: 28px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; }}
    .header {{ background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 32px; border-radius: 12px; margin-bottom: 28px; }}
    .header h1 {{ margin: 0; font-size: 28px; font-weight: 800; }}
    .header p {{ margin: 8px 0 0 0; opacity: 0.9; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .kpi-box {{ background: white; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }}
    .kpi-num {{ font-size: 32px; font-weight: 800; color: #4f46e5; margin-bottom: 4px; }}
    .kpi-title {{ font-size: 13px; text-transform: uppercase; color: #6b7280; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #f3f4f6; }}
    th {{ background: #f8fafc; color: #475569; font-weight: 600; text-transform: uppercase; font-size: 12px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
    .badge-now {{ background: #fee2e2; color: #b91c1c; }}
    .badge-next {{ background: #fef3c7; color: #b45309; }}
    .badge-later {{ background: #e0f2fe; color: #0369a1; }}
    .quote-box {{ background: #f8fafc; border-left: 4px solid #4f46e5; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; font-style: italic; font-size: 13px; }}
</style>
</head>
<body>

<div class="header">
    <h1>🚀 Executive Product Strategy & Feedback Intelligence</h1>
    <p>Comprehensive Report generated on {now_str} • Executive Decision Brief</p>
</div>

<div class="kpi-grid">
    <div class="kpi-box">
        <div class="kpi-num">{total_feedback:,}</div>
        <div class="kpi-title">Customer Requests</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-num">{len(ranking_df)}</div>
        <div class="kpi-title">Identified Themes</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-num" style="color: #ef4444;">{neg_count}</div>
        <div class="kpi-title">Friction Points</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-num" style="color: #10b981;">{top_rice}</div>
        <div class="kpi-title">Top RICE Score</div>
    </div>
</div>

<div class="card">
    <h2>🎯 Top Ranked Features (RICE Prioritization)</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Theme</th>
                <th>RICE Score</th>
                <th>Reach (Users)</th>
                <th>Impact</th>
                <th>Confidence</th>
                <th>Effort</th>
            </tr>
        </thead>
        <tbody>
"""
    for i, row in ranking_df.head(8).iterrows():
        html += f"""
            <tr>
                <td>#{i+1}</td>
                <td><strong>{row['Dominant Category']}</strong></td>
                <td><strong style="color: #4f46e5;">{row['RICE Score']}</strong></td>
                <td>{row['Reach']:,}</td>
                <td>{row['Impact']}</td>
                <td>{int(row['Confidence']*100)}%</td>
                <td>{row['Effort']} pts</td>
            </tr>
"""
    html += """
        </tbody>
    </table>
</div>

<div class="card">
    <h2>🗺️ Executive Roadmap (Now, Next, Later)</h2>
    <table>
        <thead>
            <tr>
                <th>Horizon</th>
                <th>Theme</th>
                <th>RICE</th>
                <th>Effort</th>
                <th>Assigned Team</th>
                <th>Executive Summary</th>
            </tr>
        </thead>
        <tbody>
"""
    for _, row in roadmap_df.head(10).iterrows():
        b_class = "badge-now" if row.get("Stage") == "NOW" else ("badge-next" if row.get("Stage") == "NEXT" else "badge-later")
        html += f"""
            <tr>
                <td><span class="badge {b_class}">{row.get("Stage", "PLAN")}</span></td>
                <td><strong>{row['Theme']}</strong></td>
                <td>{row['Score']}</td>
                <td>{row['Effort']} pts</td>
                <td>{row['Assigned Team']}</td>
                <td>{row['Summary'][:90]}...</td>
            </tr>
"""
    html += """
        </tbody>
    </table>
</div>

<div class="card">
    <h2>💡 Evidence-Backed Key Recommendations</h2>
"""
    for ins in insights[:4]:
        html += f"""
        <div style="margin-bottom: 20px; border-bottom: 1px solid #f3f4f6; padding-bottom: 16px;">
            <h3>{ins['Rationale_Badge']}: {ins['Theme']}</h3>
            <p><strong>AI Rationale:</strong> {ins['AI_Rationale']}</p>
            <p><strong>Action Recommendation:</strong> {ins['Actionable_Recommendation']}</p>
            <p style="font-size: 13px; color: #6b7280;">Supported by <strong>{ins['Request_Count']} requests</strong> ({ins['Pct_Of_Total']}% share) • {ins['Segment_Summary']}</p>
        """
        for q in ins['Customer_Quotes'][:2]:
            html += f"""<div class="quote-box">"{q['text']}" — <span style="color: #6b7280;">{q['segment']} user via {q['source']}</span></div>"""
        html += "</div>"

    html += """
</div>
</body>
</html>
"""
    return html

def generate_pdf_report(df, ranking_df, roadmap_df, insights, trends_df=None, segment_df=None):
    """
    Generates a professional executive PDF report using ReportLab.
    Returns bytes buffer ready for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#1e1b4b'), spaceAfter=8)
    subtitle_style = ParagraphStyle('ReportSub', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#6b7280'), spaceAfter=14)
    h2_style = ParagraphStyle('ReportH2', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor('#4f46e5'), spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#374151'))
    quote_style = ParagraphStyle('ReportQuote', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#4b5563'), leftIndent=12, italic=True)
    
    elements = []
    
    now_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph("🚀 AI Product Strategy & Feedback Intelligence Report", title_style))
    elements.append(Paragraph(f"Executive Decision Brief • Generated on {now_str} • Confidential", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4f46e5'), spaceAfter=12))
    
    # 1. KPIs
    neg_count = len(df[df['Sentiment'] == 'NEGATIVE']) if 'Sentiment' in df.columns else 0
    top_rice = ranking_df['RICE Score'].max() if not ranking_df.empty else 0
    kpi_data = [
        [
            Paragraph(f"<b>Total Requests</b><br/><font size=14 color='#4f46e5'>{len(df):,}</font>", body_style),
            Paragraph(f"<b>Identified Themes</b><br/><font size=14 color='#4f46e5'>{len(ranking_df)}</font>", body_style),
            Paragraph(f"<b>Friction Points</b><br/><font size=14 color='#ef4444'>{neg_count}</font>", body_style),
            Paragraph(f"<b>Top RICE Score</b><br/><font size=14 color='#10b981'>{top_rice}</font>", body_style)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[130, 130, 130, 130])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 14))
    
    # 2. RICE Table
    elements.append(Paragraph("🎯 Top Feature Themes by RICE Score", h2_style))
    rice_table_data = [["Rank", "Theme / Category", "RICE", "Reach", "Impact", "Conf.", "Effort"]]
    for i, row in ranking_df.head(6).iterrows():
        rice_table_data.append([
            f"#{i+1}",
            str(row['Dominant Category'])[:22],
            str(row['RICE Score']),
            f"{row['Reach']:,}",
            str(row['Impact']),
            f"{int(row['Confidence']*100)}%",
            f"{row['Effort']} pts"
        ])
    t_rice = Table(rice_table_data, colWidths=[40, 180, 55, 75, 55, 55, 60])
    t_rice.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    elements.append(t_rice)
    elements.append(Spacer(1, 14))
    
    # 3. Roadmap Table
    elements.append(Paragraph("🗺️ Now, Next, Later Roadmap", h2_style))
    roadmap_table_data = [["Horizon", "Theme", "RICE", "Effort", "Team", "Summary"]]
    for _, row in roadmap_df.head(8).iterrows():
        roadmap_table_data.append([
            str(row.get("Stage", "PLAN")),
            str(row['Theme'])[:18],
            str(row['Score']),
            f"{row['Effort']} pts",
            str(row['Assigned Team'])[:14],
            str(row['Summary'])[:40] + "..."
        ])
    t_road = Table(roadmap_table_data, colWidths=[50, 110, 45, 55, 70, 190])
    t_road.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    elements.append(t_road)
    elements.append(Spacer(1, 14))
    
    # 4. Evidence Insights
    elements.append(Paragraph("💡 Key Evidence-Backed Recommendations", h2_style))
    for ins in insights[:3]:
        elements.append(Paragraph(f"<b>{ins['Rationale_Badge']}: {ins['Theme']}</b> (RICE: {ins['RICE_Score']})", body_style))
        elements.append(Paragraph(f"<i>AI Rationale:</i> {ins['AI_Rationale']}", body_style))
        elements.append(Paragraph(f"<i>Recommendation:</i> <b>{ins['Actionable_Recommendation']}</b>", body_style))
        if ins['Customer_Quotes']:
            q = ins['Customer_Quotes'][0]
            elements.append(Paragraph(f"Customer Quote: \"{q['text'][:120]}...\" ({q['segment']} user)", quote_style))
        elements.append(Spacer(1, 6))
        
    doc.build(elements)
    buffer.seek(0)
    return buffer
