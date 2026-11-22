import pandas as pd
import re
from datetime import datetime

def load_and_clean_data(file_path_or_buffer):
    """
    Loads data from a CSV and performs robust cleaning and standardization.
    Gracefully maps diverse column headers to standard canonical names:
    - Request_Text, Category, Upvotes, Date, Request_ID, Customer_Segment, Source, Status
    """
    if isinstance(file_path_or_buffer, str):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_csv(file_path_or_buffer)
        
    if df.empty:
        raise ValueError("The provided CSV file contains no data.")

    # 1. Normalize Column Names (strip whitespace, lowercase for comparison)
    col_mapping = {col: str(col).strip().lower() for col in df.columns}
    
    # Identify the core "Text" column
    text_col = None
    text_candidates = [
        'request_text', 'text', 'request', 'feedback', 'description', 
        'content', 'feature', 'feature_request', 'summary', 'title', 'issue'
    ]
    for col, lower_col in col_mapping.items():
        if lower_col in text_candidates:
            text_col = col
            break
            
    if not text_col:
        # Fallback: find first object/string column with avg length > 10
        for col in df.columns:
            if df[col].dtype == object and df[col].dropna().astype(str).str.len().mean() > 8:
                text_col = col
                break
                
    if not text_col:
        raise ValueError(
            "Could not find a column for the Feature Request text. "
            "Please ensure your CSV has a column named 'Request_Text', 'Feedback', 'Description', or 'Feature'."
        )
        
    df = df.rename(columns={text_col: 'Request_Text'})
    
    # Identify and rename other columns
    for col, lower_col in col_mapping.items():
        if col == text_col:
            continue
        if lower_col in ['category', 'type', 'tag', 'component', 'module']:
            df = df.rename(columns={col: 'Category'})
        elif lower_col in ['upvotes', 'votes', 'score', 'likes', 'count']:
            df = df.rename(columns={col: 'Upvotes'})
        elif lower_col in ['date', 'timestamp', 'created_at', 'submitted', 'time']:
            df = df.rename(columns={col: 'Date'})
        elif lower_col in ['id', 'request_id', 'ticket_id', 'feedback_id']:
            df = df.rename(columns={col: 'Request_ID'})
        elif lower_col in ['customer_segment', 'segment', 'tier', 'plan', 'user_tier', 'account_type']:
            df = df.rename(columns={col: 'Customer_Segment'})
        elif lower_col in ['source', 'channel', 'origin', 'platform']:
            df = df.rename(columns={col: 'Source'})
        elif lower_col in ['status', 'state', 'stage']:
            df = df.rename(columns={col: 'Status'})

    # 2. Add missing columns with intelligent defaults
    if 'Category' not in df.columns:
        df['Category'] = "General"
    else:
        df['Category'] = df['Category'].fillna("General").astype(str)

    if 'Upvotes' not in df.columns:
        df['Upvotes'] = 1
    else:
        df['Upvotes'] = pd.to_numeric(df['Upvotes'], errors='coerce').fillna(1).astype(int)

    if 'Request_ID' not in df.columns:
        df['Request_ID'] = [f"REQ-{i+1:04d}" for i in range(len(df))]

    if 'Date' not in df.columns:
        df['Date'] = datetime.today().strftime('%Y-%m-%d')
    else:
        # Standardize date format to YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today()).dt.strftime('%Y-%m-%d')

    if 'Customer_Segment' not in df.columns:
        # Heuristic assignment based on Upvotes
        def estimate_segment(upvotes):
            if upvotes > 50:
                return "Enterprise"
            elif upvotes > 20:
                return "Pro"
            elif upvotes > 5:
                return "Starter"
            return "Free"
        df['Customer_Segment'] = df['Upvotes'].apply(estimate_segment)
    else:
        df['Customer_Segment'] = df['Customer_Segment'].fillna("Pro").astype(str)

    if 'Source' not in df.columns:
        df['Source'] = "In-App Feedback"
    else:
        df['Source'] = df['Source'].fillna("In-App Feedback").astype(str)

    if 'Status' not in df.columns:
        df['Status'] = "Backlog"
    else:
        df['Status'] = df['Status'].fillna("Backlog").astype(str)

    # 3. Text cleaning
    df = df.dropna(subset=['Request_Text'])
    df['Cleaned_Text'] = df['Request_Text'].apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', str(x))).strip()
    )
    # Remove empty cleaned texts
    df = df[df['Cleaned_Text'].str.len() > 0].reset_index(drop=True)
    
    return df
