import pandas as pd
import random
import os
from datetime import datetime, timedelta

# Categories
CATEGORIES = [
    "Authentication", "Payments", "Dashboard", "Notifications", 
    "Reports", "Mobile App", "UI/UX", "Performance", "API", "Security"
]

CUSTOMER_SEGMENTS = ["Enterprise", "Pro", "Starter", "Free"]
SEGMENT_WEIGHTS = [0.25, 0.35, 0.20, 0.20]

SOURCES = ["Intercom", "Zendesk", "Sales Call", "G2 Review", "Slack Community", "In-App Feedback"]
SOURCE_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]

STATUSES = ["Backlog", "In Review", "Planned", "In Progress"]
STATUS_WEIGHTS = [0.55, 0.25, 0.12, 0.08]

PROMPTS = {
    "Authentication": [
        "Please add Two-Factor Authentication via Authenticator App.",
        "We need SSO integration with Google Workspace, Okta, and Microsoft Azure AD.",
        "Passwordless magic link login would drastically improve onboarding.",
        "Can we have biometric face/fingerprint login for mobile users?",
        "Implement OAuth2 and SAML 2.0 for enterprise identity management.",
        "Users should be required to enforce session timeouts and MFA."
    ],
    "Payments": [
        "Payment integration with Stripe Elements and Apple Pay is missing.",
        "Please add PayPal and SEPA direct debit support for European clients.",
        "I want to export itemized tax invoices directly as compliant PDFs.",
        "Multi-currency billing and automated VAT calculation at checkout.",
        "Crypto payment gateway integration with USDT and Bitcoin.",
        "Automated subscription dunning and prorated upgrade receipts."
    ],
    "Dashboard": [
        "Add a high-contrast dark mode toggle across all dashboard views.",
        "Customizable dashboard layout where widgets can be rearranged.",
        "Drag and drop KPI cards and charts on the home overview screen.",
        "The dashboard should stream real-time telemetry metrics via websockets.",
        "Executive summary top banner showing monthly recurring revenue and churn.",
        "Make the main dashboard responsive and legible on iPad and tablets."
    ],
    "Notifications": [
        "Push notifications for critical alerts on both iOS and Android.",
        "Granular notification preferences to control email digest frequency.",
        "Slack and Microsoft Teams webhook alerts for real-time ticket triggers.",
        "In-app notification bell center with unread counters and click-to-open.",
        "SMS emergency alerts for high-priority operational downtime.",
        "Smart alert batching to avoid blowing up user inboxes with spam."
    ],
    "Reports": [
        "One-click Export to Excel (.xlsx) with formulas and styling intact.",
        "Automated scheduled weekly PDF reports delivered to executive leadership.",
        "Custom rolling date range selectors (last 7, 30, 90, 365 days).",
        "Bulk CSV data export for data warehouse pipelines and BI tools.",
        "Interactive cohort retention heatmaps in the analytics report.",
        "Granular drill-down filtering by department and sales territory."
    ],
    "Mobile App": [
        "Offline caching mode so field reps can view customer details without signal.",
        "Android app crashes immediately on launch when opening large projects.",
        "Optimized split-screen tablet interface for Apple iPad and Galaxy Tab.",
        "Biometric FaceID authentication on mobile app start.",
        "Bottom navigation bar is awkward to reach with one hand on larger phones.",
        "Background data synchronization to prevent stale information on mobile."
    ],
    "UI/UX": [
        "The text contrast on table headers is too faint and fails WCAG AA standards.",
        "Make the primary 'Save Changes' button more prominent and fixed in view.",
        "Adopt a modern, legible sans-serif font family with better font weights.",
        "Interactive contextual walkthrough tooltips for newly invited team members.",
        "Reduce multi-step wizards from 5 clicks down to 2 straightforward steps.",
        "Collapsible sidebar navigation to maximize workspace canvas space."
    ],
    "Performance": [
        "First contentful paint takes over 4 seconds on initial workspace load.",
        "Full-text search queries take 3+ seconds to return matching results.",
        "Compress and serve asset images in WebP format to reduce data usage.",
        "Memory leak in browser tab when leaving dashboard open overnight.",
        "Optimize database queries and add Redis caching to lower p99 latency.",
        "Implement virtualized scrolling for lists containing over 500 items."
    ],
    "API": [
        "Provide a GraphQL API endpoint for querying nested relational entities.",
        "Public API rate limits (60 req/min) are too restrictive for enterprise sync.",
        "Configurable webhooks with HMAC signature verification for security.",
        "Return structured RFC-7807 problem details in API error responses.",
        "Interactive OpenAPI / Swagger documentation with runnable curl snippets.",
        "Batch mutation endpoint for updating up to 1,000 records in a single call."
    ],
    "Security": [
        "End-to-end encryption for team messages and sensitive document attachments.",
        "Comprehensive immutable audit logs tracking all admin and user actions.",
        "Granular Role-Based Access Control (RBAC) with custom role definitions.",
        "IP address whitelisting and CIDR block restrictions for API tokens.",
        "Configurable idle session automatic logout after 15 or 30 minutes.",
        "PII data masking and pseudonymization in export files and audit logs."
    ]
}

def generate_variations(text):
    prefixes = [
        "", "We urgently need ", "Please consider adding ", "It would be fantastic if ",
        "Our team requires ", "Can we please have ", "Crucial request: ", "As a customer, I need "
    ]
    suffixes = [
        "", " as soon as possible.", " for our upcoming deployment.", " to unblock our team.",
        " in the next quarterly release.", ".", " — this is a huge blocker for us."
    ]
    
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    base = text.lower() if prefix else text
    if random.random() < 0.15:
        base = base.replace(".", "")
    return f"{prefix}{base}{suffix}".strip()

def main():
    print("Generating 1,000 rich, realistic sample feature requests...")
    random.seed(42)
    
    # Generate dates across past 12 months, with recent momentum in Security, API, and Mobile
    now = datetime(2026, 11, 20)
    data = []
    
    for i in range(1, 1001):
        category = random.choice(CATEGORIES)
        base_prompt = random.choice(PROMPTS[category])
        request_text = generate_variations(base_prompt)
        user_id = f"USR-{random.randint(1000, 9999)}"
        
        # Upvotes: Power-law like distribution (a few high upvote features)
        if random.random() < 0.1:
            upvotes = random.randint(80, 250)
        elif random.random() < 0.3:
            upvotes = random.randint(30, 80)
        else:
            upvotes = random.randint(2, 30)
            
        # Trending bias: make Security & API more frequent in recent 60 days
        if category in ["Security", "API", "Mobile App"] and random.random() < 0.6:
            days_ago = random.randint(1, 60)
        else:
            days_ago = random.randint(1, 365)
            
        req_date = (now - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        customer_segment = random.choices(CUSTOMER_SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        source = random.choices(SOURCES, weights=SOURCE_WEIGHTS)[0]
        status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        
        data.append({
            "Request_ID": f"REQ-{i:04d}",
            "User_ID": user_id,
            "Date": req_date,
            "Category": category,
            "Customer_Segment": customer_segment,
            "Source": source,
            "Status": status,
            "Request_Text": request_text,
            "Upvotes": upvotes
        })
        
    df = pd.DataFrame(data)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "sample_requests.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} requests and saved to {output_path}")

if __name__ == "__main__":
    main()
