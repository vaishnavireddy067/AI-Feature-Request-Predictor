import os
import pandas as pd

def get_agent(hf_token: str = None, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    """
    Initializes a smolagents CodeAgent powered by the Hugging Face Inference API.
    """
    if not hf_token:
        return None
    try:
        from smolagents import CodeAgent, InferenceClientModel
        os.environ["HF_TOKEN"] = hf_token
        model = InferenceClientModel(model_id=model_id)
        agent = CodeAgent(
            tools=[], 
            model=model, 
            additional_authorized_imports=["pandas", "numpy", "math", "datetime"]
        )
        return agent
    except Exception as e:
        print(f"Error initializing smolagents: {e}")
        return None

def query_agent(agent, query: str, df: pd.DataFrame, roadmap_df: pd.DataFrame):
    """
    Executes an analytical question against customer feedback and roadmap datasets.
    """
    if agent is None:
        # Smart local analytical fallback
        q_lower = query.lower()
        if "quick win" in q_lower or "quick" in q_lower:
            quick_wins = roadmap_df[(roadmap_df['Effort'] <= 4.0) & (roadmap_df['Score'] >= roadmap_df['Score'].median())]
            if not quick_wins.empty:
                themes = ", ".join(f"**{t}** (Score: {s}, Effort: {e})" for t, s, e in zip(quick_wins['Theme'], quick_wins['Score'], quick_wins['Effort']))
                return f"🔍 **AI Analysis (Local Engine):**\n\nI identified {len(quick_wins)} high-leverage quick wins (Low Effort ≤ 4 pts, High RICE Score):\n- {themes}\n\n**Recommendation:** Prioritize these in your immediate upcoming sprint for fast customer impact."
        if "negative" in q_lower or "friction" in q_lower or "churn" in q_lower:
            high_neg = roadmap_df[roadmap_df['Negative_Sentiment_Pct'] >= 30.0].sort_values(by='Negative_Sentiment_Pct', ascending=False)
            if not high_neg.empty:
                themes = "\n".join(f"- **{t}**: {n}% negative feedback ({v} requests)" for t, n, v in zip(high_neg['Theme'], high_neg['Negative_Sentiment_Pct'], high_neg['Volume']))
                return f"🚨 **High Friction Themes (Local Engine):**\n\nThese features have the highest customer dissatisfaction:\n{themes}\n\n**Recommendation:** Direct focus here to reduce customer churn."
        return f"💡 **Dataset Overview (Local Engine):**\n\n- Total Requests: {len(df):,}\n- Total Unique Themes: {len(roadmap_df)}\n- Top Ranked Theme: **{roadmap_df.iloc[0]['Theme']}** with RICE score {roadmap_df.iloc[0]['Score']}.\n\n*Note: Connect a free Hugging Face token in the sidebar for full conversational Python code execution via Qwen-2.5.*"

    os.makedirs("data", exist_ok=True)
    df_path = "data/temp_features.csv"
    roadmap_path = "data/temp_roadmap.csv"
    
    df.to_csv(df_path, index=False)
    roadmap_df.to_csv(roadmap_path, index=False)
    
    prompt = (
        f"You are an expert AI Product Manager and Data Analyst. "
        f"You have access to two CSV files:\n"
        f"1. `data/temp_features.csv`: Contains the raw feature requests and their clusters/sentiments.\n"
        f"2. `data/temp_roadmap.csv`: Contains the aggregated prioritization roadmap (Themes, Priority, Score, Effort).\n\n"
        f"Please write and execute python code using pandas to analyze these files and answer the following question:\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Provide a clear, detailed, and nicely formatted markdown response."
    )
    
    try:
        result = agent.run(prompt)
        return str(result)
    except Exception as e:
        return f"❌ An error occurred while running the agent: {str(e)}"

def generate_prd(hf_token: str, theme_name: str, summary: str, keywords: str, sample_requests: str, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    """
    Acts as an AI Product Manager to generate a Product Requirements Document (PRD).
    Includes intelligent offline template fallback.
    """
    if hf_token:
        try:
            from smolagents import InferenceClientModel
            os.environ["HF_TOKEN"] = hf_token
            model = InferenceClientModel(model_id=model_id)
            messages = [
                {"role": "system", "content": "You are an expert AI Product Manager. Write a professional, actionable Product Requirement Document (PRD)."},
                {"role": "user", "content": f"Write a PRD for the feature theme '{theme_name}'.\n\nContext:\n- Summary: {summary}\n- Keywords: {keywords}\n- Top Feedback Samples: {sample_requests}\n\nPlease include:\n1. Goal / Problem Statement\n2. User Stories\n3. Key Requirements & Acceptance Criteria\n4. Potential Technical Considerations"}
            ]
            response = model(messages)
            return response.content
        except Exception:
            pass
            
    # Deterministic Local PRD Template Fallback
    return f"""# 📄 Product Requirements Document (PRD)
## Feature Theme: {theme_name}

### 1. Goal / Problem Statement
Customers are encountering significant friction and requesting structured capabilities around **{theme_name}**.
> *Context Summary:* {summary}
> *Core Themes/Keywords:* {keywords}

### 2. User Stories
- **As an** Enterprise Administrator, **I want** robust and compliant {theme_name.lower()} controls **so that** our organization meets compliance standards.
- **As a** regular user, **I want** an intuitive, seamless workflow for {theme_name.lower()} **so that** I can save time without encountering errors.
- **As a** developer or integration partner, **I want** reliable API and data consistency **so that** automated workflows do not break.

### 3. Key Requirements & Acceptance Criteria
- [ ] **Functional:** Implement core workflow covering the most frequent user feedback requests.
- [ ] **Error Handling:** Clear, contextual error states with actionable user recovery guidance.
- [ ] **Performance:** Response latency for operations must remain under 300ms at p95.
- [ ] **Analytics:** Instrument telemetry events for feature adoption, failure rate, and daily active usage.

### 4. Technical Considerations & Architecture
- Data model schema migrations and backwards-compatibility for existing accounts.
- Cache invalidation strategy and index optimization for high-concurrency access.
- Security audit: Ensure role-based authorization (RBAC) checks are enforced at API layer.
"""

def generate_release_notes(hf_token: str, theme_name: str, priority: str, effort: str, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    """
    Generates customer-facing release notes with offline fallback.
    """
    if hf_token:
        try:
            from smolagents import InferenceClientModel
            os.environ["HF_TOKEN"] = hf_token
            model = InferenceClientModel(model_id=model_id)
            messages = [
                {"role": "system", "content": "You are an expert AI Product Marketing Manager. Write exciting, engaging release notes."},
                {"role": "user", "content": f"Write an exciting public release note for: '{theme_name}'.\nContext:\n- Internal Priority: {priority}\n- Development Effort: {effort} pts\nInclude: Catchy headline, What's new & why users will love it, and How it works."}
            ]
            response = model(messages)
            return response.content
        except Exception:
            pass
            
    return f"""# 🎉 What's New: Introducing {theme_name}
### Turbocharge your workflow with our latest release!

We've listened closely to your feedback! Today, we're thrilled to unveil major enhancements to **{theme_name}**.

#### 🌟 Why You'll Love It
- **Effortless Productivity:** Designed from the ground up based on real customer feedback to eliminate daily friction.
- **Built for Scale:** Engineered for speed, reliability, and enterprise-grade performance.
- **Seamless Integration:** Works out of the box with your existing configurations and data.

#### ⚙️ How It Works
1. Navigate to the **{theme_name}** section in your dashboard.
2. Configure your custom preferences in just a couple of clicks.
3. Enjoy an upgraded, uninterrupted experience!

*Have feedback on this release? Drop us a note via in-app chat — we're constantly building for you!*
"""
