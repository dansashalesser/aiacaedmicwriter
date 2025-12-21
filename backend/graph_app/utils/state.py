"""State definition for the graph."""

from typing import TypedDict, Optional, Dict, List


class GraphState(TypedDict):
    """
    State for the graph.
    Add more fields as needed for your application.
    """
    # Scholar Extraction
    query: str  # User's search query
    papers: Optional[List[Dict]]  # List of papers from Semantic Scholar
    paper_count: int  # Number of papers retrieved
    formatted_results: Optional[str]  # Formatted text response
    json_path: Optional[str]  # Path to saved JSON file
