import sys
print("Python Executable:", sys.executable)
try:
    from utils.ai_agent import get_agent, query_agent, generate_prd, generate_release_notes
    from utils.preprocessing import load_and_clean_data
    from utils.ranking import rank_features, calculate_rice_score
    from utils.roadmap import generate_ai_roadmap
    from utils.insights import generate_evidence_backed_insights
    from utils.trends import analyze_demand_trends
    from utils.segments import analyze_customer_segments
    from utils.reporting import generate_markdown_report, generate_html_report, generate_pdf_report
    print("SUCCESS: All utils modules imported successfully!")
except Exception as e:
    import traceback
    print("FAILED TO IMPORT")
    traceback.print_exc()
