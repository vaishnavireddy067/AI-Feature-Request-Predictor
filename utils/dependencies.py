import pandas as pd

# Knowledge graph of SaaS architectural dependencies
DOMAIN_DEPENDENCIES = {
    "Security": {
        "prerequisites": ["Authentication"],
        "enables": ["API", "Compliance", "Enterprise Governance"],
        "reason": "Enterprise role-based access and audit logs require core user identity & authentication."
    },
    "API": {
        "prerequisites": ["Authentication", "Security"],
        "enables": ["Developer Webhooks", "Integrations"],
        "reason": "Secure public endpoints require API token authentication and rate limiting."
    },
    "Reports": {
        "prerequisites": ["Dashboard"],
        "enables": ["Executive Telemetry", "Scheduled Exports"],
        "reason": "Aggregated reporting queries build on dashboard data pipelines."
    },
    "Mobile App": {
        "prerequisites": ["Authentication", "API"],
        "enables": ["Field Operations", "Push Alerts"],
        "reason": "Mobile clients require authentication and backend REST/GraphQL endpoints."
    },
    "Notifications": {
        "prerequisites": ["Authentication"],
        "enables": ["Slack / Webhook Integrations", "Real-time Alerts"],
        "reason": "Personalized alerts require user notification preference settings."
    },
    "Payments": {
        "prerequisites": ["Authentication", "Security"],
        "enables": ["Multi-currency Billing", "Invoicing"],
        "reason": "Payment gateway integrations require PCI-compliant encrypted tokens and user identity."
    },
    "Performance": {
        "prerequisites": [],
        "enables": ["Dashboard", "API", "Mobile App"],
        "reason": "Foundational caching (Redis) and database indexing directly accelerates all modules."
    },
    "Authentication": {
        "prerequisites": [],
        "enables": ["Security", "API", "Payments", "Notifications", "Mobile App"],
        "reason": "Core foundational identity tier for all user-facing services."
    },
    "Dashboard": {
        "prerequisites": ["Authentication"],
        "enables": ["Reports", "Executive Overview"],
        "reason": "User metrics dashboard requires authenticated tenant sessions."
    },
    "UI/UX": {
        "prerequisites": [],
        "enables": ["Dashboard", "Mobile App", "Accessibility"],
        "reason": "Design system and styling tokens inform all frontend interfaces."
    }
}

def map_feature_dependencies(roadmap_df):
    """
    Analyzes feature themes in the roadmap and generates a dependency mapping:
    - Lists prerequisites (what this feature needs first)
    - Lists downstream dependents (what this feature unblocks)
    - Checks for roadmap order violations (e.g. feature in NOW depending on a prerequisite in LATER)
    """
    dependency_records = []
    
    # Map each theme to its current stage (NOW, NEXT, LATER)
    theme_to_stage = {}
    for _, row in roadmap_df.iterrows():
        theme_name = str(row['Theme'])
        stage = str(row.get('Stage', 'PLAN'))
        theme_to_stage[theme_name.lower()] = stage
        
    stage_weights = {"NOW": 0, "NEXT": 1, "LATER": 2, "PLAN": 1}

    for _, row in roadmap_df.iterrows():
        theme = str(row['Theme'])
        current_stage = str(row.get('Stage', 'NOW'))
        current_weight = stage_weights.get(current_stage, 1)
        
        prereqs = []
        enables = []
        reason = "Independent feature capability with no blocking dependencies."
        
        # Match domain rule
        for domain, meta in DOMAIN_DEPENDENCIES.items():
            if domain.lower() in theme.lower():
                prereqs = meta["prerequisites"]
                enables = meta["enables"]
                reason = meta["reason"]
                break
                
        # Check blocker status
        is_blocked = False
        blocking_items = []
        for prereq in prereqs:
            for known_theme, prereq_stage in theme_to_stage.items():
                if prereq.lower() in known_theme:
                    prereq_weight = stage_weights.get(prereq_stage, 1)
                    # If this feature is scheduled before its prerequisite, it's blocked!
                    if current_weight < prereq_weight:
                        is_blocked = True
                        blocking_items.append(f"{prereq} (currently in {prereq_stage})")
                        
        if is_blocked:
            status = f"⚠️ Blocked by {', '.join(blocking_items)}"
            status_badge = "Blocked"
        elif prereqs:
            status = f"✅ Prerequisites Ready: {', '.join(prereqs)}"
            status_badge = "Ready"
        else:
            status = "🚀 Foundational (No Prerequisites)"
            status_badge = "Foundational"
            
        dependency_records.append({
            "Theme": theme,
            "Current_Stage": current_stage,
            "Prerequisites": ", ".join(prereqs) if prereqs else "None",
            "Downstream_Enables": ", ".join(enables) if enables else "None",
            "Dependency_Status": status,
            "Status_Badge": status_badge,
            "Architectural_Rationale": reason,
            "Is_Blocked": is_blocked
        })
        
    dep_df = pd.DataFrame(dependency_records)
    return dep_df
