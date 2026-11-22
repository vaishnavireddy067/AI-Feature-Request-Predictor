import re

# Simple heuristic mapping for auto-tagging
TEAM_TAGS = {
    "Mobile": ["app", "ios", "android", "mobile", "tablet", "phone", "smartphone", "iphone", "ipad"],
    "Frontend": ["ui", "button", "screen", "page", "dashboard", "layout", "color", "dark mode", "font", "web", "browser", "click", "menu", "frontend", "interface"],
    "Backend": ["slow", "crash", "load", "error", "server", "timeout", "offline", "sync", "database", "backend", "performance"],
    "Billing": ["invoice", "price", "pay", "credit card", "subscription", "plan", "upgrade", "downgrade", "billing", "receipt", "charge"],
    "API / Integrations": ["api", "webhook", "integration", "zapier", "slack", "export", "import", "developer", "endpoint"],
    "Security": ["login", "password", "2fa", "auth", "security", "permission", "access", "hacked", "breach", "sso"]
}

def auto_tag_request(text):
    """
    Scans a request text and returns a list of assigned team tags based on keyword heuristics.
    """
    text_lower = str(text).lower()
    assigned_tags = []
    
    for team, keywords in TEAM_TAGS.items():
        if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in keywords):
            assigned_tags.append(team)
            
    if not assigned_tags:
        assigned_tags.append("General")
        
    return assigned_tags

def tag_dataframe(df, text_col='Cleaned_Text'):
    """
    Applies auto-tagging to an entire DataFrame.
    """
    if text_col in df.columns:
        df['Auto_Tags'] = df[text_col].apply(auto_tag_request)
    return df
