import pytest
import pandas as pd
from utils.visualization import (
    plot_interactive_rice_matrix,
    plot_interactive_trends,
    plot_segment_donut,
    plot_target_demand_trend
)
from utils.simulator import plot_scenario_comparison_chart
from utils.capacity import plot_capacity_utilization_chart

def test_rice_matrix_layout_no_collisions():
    roadmap_mock = pd.DataFrame({
        "Theme": ["AI Dashboard", "Tax Invoicing", "WCAG Theme"],
        "Effort": [4.0, 7.0, 3.5],
        "Score": [45.2, 85.0, 62.1],
        "Volume": [120, 80, 50],
        "Stage": ["NOW", "NEXT", "LATER"],
        "Assigned Team": ["Growth", "Platform", "Design"],
        "Summary": ["Feature 1", "Feature 2", "Feature 3"]
    })
    fig = plot_interactive_rice_matrix(roadmap_mock)
    
    # 1. Left margin must be >= 70 to prevent y-axis label clipping
    assert fig.layout.margin.l >= 70, f"Left margin too small: {fig.layout.margin.l}"
    
    # 2. Top margin must be >= 50 for title, and bottom margin >= 50 for horizontal legend
    assert fig.layout.margin.t >= 50, f"Top margin too small: {fig.layout.margin.t}"
    assert fig.layout.margin.b >= 50, f"Bottom margin too small: {fig.layout.margin.b}"
    
    # 3. Title positioned on left
    assert fig.layout.title.x <= 0.05
    assert fig.layout.title.y >= 0.95
    
    # 4. Legend horizontal and placed cleanly below the plot area
    assert fig.layout.legend.orientation == 'h'
    assert fig.layout.legend.y <= 0.0
    
    # 5. Y-axis headroom covers max_score * 1.2
    max_score = roadmap_mock['Score'].max()
    assert fig.layout.yaxis.range[1] >= max_score * 1.2

def test_trends_layout_no_collisions():
    df_mock = pd.DataFrame({
        "Date": ["2026-01-15", "2026-02-15", "2026-03-15"],
        "Sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "Customer_Segment": ["Enterprise", "Pro", "Free"]
    })
    fig = plot_interactive_trends(df_mock)
    
    assert fig.layout.margin.l >= 65, f"Left margin too small: {fig.layout.margin.l}"
    assert fig.layout.margin.t >= 50, f"Top margin too small: {fig.layout.margin.t}"
    assert fig.layout.margin.b >= 50, f"Bottom margin too small: {fig.layout.margin.b}"
    assert fig.layout.title.x <= 0.05
    assert fig.layout.legend.orientation == 'h'
    assert fig.layout.legend.y <= 0.0

def test_scenario_comparison_chart_layout():
    comp_df = pd.DataFrame({
        "Theme": ["AI Feature", "Export CSV"],
        "Score_A_Norm": [8.5, 4.2],
        "Score_B_Norm": [9.1, 5.0]
    })
    fig = plot_scenario_comparison_chart(comp_df, "Scenario A", "Scenario B")
    
    assert fig.layout.margin.l >= 150
    assert fig.layout.legend.orientation == 'h'
    assert fig.layout.legend.y <= 0.0

def test_capacity_utilization_chart_layout():
    summary_mock = pd.DataFrame({
        "Sprint": ["Sprint 1", "Sprint 2"],
        "Allocated_Points": [35.0, 48.0],
        "Utilization_Pct": [87.5, 120.0]
    })
    fig = plot_capacity_utilization_chart(summary_mock, capacity_limit=40.0)
    
    assert fig.layout.margin.l >= 60
    assert fig.layout.margin.t >= 65
    assert "Story Points" in fig.layout.yaxis.title.text
