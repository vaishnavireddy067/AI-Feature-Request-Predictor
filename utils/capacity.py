import pandas as pd
import numpy as np
import plotly.graph_objects as go

def calculate_capacity_plan(
    roadmap_df: pd.DataFrame,
    dependency_df: pd.DataFrame = None,
    team_velocity: float = 40.0,
    sprint_duration_weeks: int = 2,
    num_engineers: int = 5,
    sprints_per_horizon: int = 3
):
    """
    Calculates engineering sprint capacity and detects overbooked sprints.
    Maps Now / Next / Later items into discrete sprint slots (e.g. Sprints 1 to 6).
    """
    df_items = roadmap_df.copy()
    
    # Standard story points conversion: Effort (1-10) * 3 = Story Points (3 to 30 pts)
    effort_col = 'Effort' if 'Effort' in df_items.columns else 'score'
    df_items['Story_Points'] = df_items[effort_col].apply(lambda x: round(float(x) * 3.0, 1))
    
    # Total Sprint Capacity = Team Velocity (Story Points per sprint)
    capacity_per_sprint = float(team_velocity)
    
    # Sprints 1-3 correspond to NOW, Sprints 4-6 to NEXT, Sprints 7-9 to LATER
    sprint_names = [f"Sprint {i+1} ({sprint_duration_weeks}w)" for i in range(sprints_per_horizon * 3)]
    
    # Assign items to sprints within their horizon
    horizon_sprints = {
        "NOW": [0, 1, 2],
        "NEXT": [3, 4, 5],
        "LATER": [6, 7, 8]
    }
    
    assigned_sprints = []
    sprint_loads = {i: 0.0 for i in range(len(sprint_names))}
    
    for _, row in df_items.iterrows():
        stage = str(row.get('Stage', 'NOW')).upper()
        available_slots = horizon_sprints.get(stage, [0, 1, 2])
        pts = float(row['Story_Points'])
        
        # Greedy assignment to earliest sprint in horizon with space or least load
        best_slot = available_slots[0]
        for slot in available_slots:
            if sprint_loads[slot] + pts <= capacity_per_sprint:
                best_slot = slot
                break
        else:
            # If all are full, pick the slot with lowest load in this horizon
            best_slot = min(available_slots, key=lambda s: sprint_loads[s])
            
        sprint_loads[best_slot] += pts
        assigned_sprints.append(sprint_names[best_slot])
        
    df_items['Assigned_Sprint'] = assigned_sprints
    
    # Build Sprint Summary DataFrame
    summary_records = []
    for i, s_name in enumerate(sprint_names[:sprints_per_horizon * 2]):  # Focus on first 6 sprints (NOW + NEXT)
        load = round(sprint_loads[i], 1)
        cap = capacity_per_sprint
        util_pct = round((load / cap) * 100, 1) if cap > 0 else 0.0
        is_overbooked = load > cap
        over_delta = round(load - cap, 1) if is_overbooked else 0.0
        
        if util_pct > 100:
            status = "🚨 Overbooked"
            badge = "Overbooked"
        elif util_pct >= 85:
            status = "🟡 Near Capacity"
            badge = "Optimal"
        else:
            status = "🟢 Healthy"
            badge = "Under Capacity"
            
        summary_records.append({
            "Sprint": s_name,
            "Sprint_Index": i + 1,
            "Horizon": "NOW (Q1)" if i < 3 else "NEXT (Q2)",
            "Allocated_Points": load,
            "Capacity_Points": cap,
            "Utilization_Pct": util_pct,
            "Status": status,
            "Badge": badge,
            "Overbooked_Delta": over_delta
        })
        
    summary_df = pd.DataFrame(summary_records)
    overbooked_count = len(summary_df[summary_df['Allocated_Points'] > summary_df['Capacity_Points']])
    
    # Check Dependency Violations across sprints
    dependency_violations = []
    if dependency_df is not None and not dependency_df.empty:
        # Map theme to sprint index
        theme_to_sprint_idx = {}
        for _, r in df_items.iterrows():
            t_name = str(r['Theme']).lower()
            s_name = str(r['Assigned_Sprint'])
            s_idx = int(s_name.split()[1]) if 'Sprint' in s_name else 1
            theme_to_sprint_idx[t_name] = s_idx
            
        for _, dep_row in dependency_df.iterrows():
            t = str(dep_row['Theme']).lower()
            prereqs = [p.strip().lower() for p in str(dep_row.get('Prerequisites', '')).split(',') if p.strip() and p.strip().lower() != 'none']
            curr_sprint = theme_to_sprint_idx.get(t, 1)
            
            for p in prereqs:
                for k_theme, p_sprint in theme_to_sprint_idx.items():
                    if p in k_theme and curr_sprint < p_sprint:
                        dependency_violations.append({
                            "Feature": dep_row['Theme'],
                            "Feature_Sprint": f"Sprint {curr_sprint}",
                            "Prerequisite": p.title(),
                            "Prereq_Sprint": f"Sprint {p_sprint}",
                            "Violation": f"'{dep_row['Theme']}' is scheduled in Sprint {curr_sprint}, but its prerequisite '{p.title()}' is scheduled later in Sprint {p_sprint}!"
                        })
                        
    return {
        "df_items": df_items,
        "summary_df": summary_df,
        "capacity_per_sprint": capacity_per_sprint,
        "overbooked_count": overbooked_count,
        "dependency_violations": dependency_violations
    }

def auto_balance_roadmap_capacity(roadmap_df: pd.DataFrame, team_velocity: float = 40.0):
    """
    Intelligently rebalances features across Now / Next / Later to ensure NO sprint is overbooked.
    Features with highest RICE priority claim earlier capacity slots.
    """
    df_copy = roadmap_df.copy()
    if 'Score' not in df_copy.columns and 'RICE Score' in df_copy.columns:
        df_copy['Score'] = df_copy['RICE Score']
        
    df_sorted = df_copy.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    # 3 sprints per horizon * team_velocity
    now_capacity = team_velocity * 3.0    # e.g. 120 points
    next_capacity = team_velocity * 3.0   # e.g. 120 points
    
    now_used = 0.0
    next_used = 0.0
    
    new_stages = []
    new_priorities = []
    new_timeframes = []
    audit_notes = []
    
    for _, row in df_sorted.iterrows():
        pts = float(row.get('Effort', 5.0)) * 3.0
        theme = row['Theme']
        old_stage = row.get('Stage', 'NOW')
        
        if now_used + pts <= now_capacity:
            now_used += pts
            new_stage = "NOW"
            prio = "🔥 NOW (Current Sprint / Q1)"
            timeframe = "1 - 3 Months"
        elif next_used + pts <= next_capacity:
            next_used += pts
            new_stage = "NEXT"
            prio = "⭐ NEXT (Upcoming Quarter / Q2)"
            timeframe = "3 - 6 Months"
        else:
            new_stage = "LATER"
            prio = "🧊 LATER (Future Horizon / Backlog)"
            timeframe = "6 - 12 Months"
            
        if old_stage != new_stage:
            audit_notes.append(f"Shifted **{theme}** from **{old_stage}** → **{new_stage}** to prevent sprint overload.")
            
        new_stages.append(new_stage)
        new_priorities.append(prio)
        new_timeframes.append(timeframe)
        
    df_sorted['Stage'] = new_stages
    df_sorted['Priority'] = new_priorities
    df_sorted['Timeframe'] = new_timeframes
    
    return df_sorted, audit_notes

def plot_capacity_utilization_chart(summary_df: pd.DataFrame, capacity_limit: float = 40.0):
    """
    Renders an interactive Plotly bar chart comparing allocated story points vs sprint capacity.
    Overbooked sprints are highlighted in coral/red, healthy sprints in purple/blue.
    """
    colors = []
    for util in summary_df['Utilization_Pct']:
        if util > 100:
            colors.append('#EF4444')  # Red alert
        elif util >= 85:
            colors.append('#F59E0B')  # Amber warning
        else:
            colors.append('#6366F1')  # Indigo healthy
            
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=summary_df['Sprint'],
        y=summary_df['Allocated_Points'],
        marker_color=colors,
        text=summary_df['Utilization_Pct'].apply(lambda x: f"<b>{x:.0f}%</b>"),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Allocated: %{y:.1f} pts<br>Capacity: ' + f'{capacity_limit:.1f} pts' + '<extra></extra>'
    ))
    
    # Capacity limit horizontal reference line
    fig.add_hline(
        y=capacity_limit,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"Max Team Capacity ({capacity_limit:.0f} pts)",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", size=11, family="Plus Jakarta Sans, sans-serif")
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Sprint Velocity & Story Point Load vs Team Capacity</b>",
            font=dict(size=14, color="#0F172A", family="Plus Jakarta Sans, sans-serif")
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor="transparent", tickfont=dict(size=11, color="#334155")),
        yaxis=dict(
            title="Story Points (Effort × 3)",
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            range=[0, max(summary_df['Allocated_Points'].max() * 1.25, capacity_limit * 1.3)]
        )
    )
    return fig
