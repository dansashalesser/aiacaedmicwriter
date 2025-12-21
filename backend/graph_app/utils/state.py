"""State definition for the graph."""

from typing import TypedDict, Optional, Dict, List


class GraphState(TypedDict):
    """
    State for the graph.
    Add more fields as needed for your application.
    """
    # NEW: User input and topic segmentation
    user_input: str  # Detailed paper idea
    topics: Optional[List[str]]  # Extracted research topics (1-5)
    max_topics: int  # Configurable limit (default 5)
    papers_per_topic: int  # Papers to fetch per topic (default 5)
    papers_for_original_query: int  # Papers to fetch for original query (default 10)

    # NEW: Categorized results
    papers_by_topic: Optional[Dict[str, List[Dict]]]  # Papers grouped by topic (includes original query)
    total_papers: int  # Sum across original query + all topics

    # EXISTING: Keep for backwards compatibility
    query: str  # User's search query
    papers: Optional[List[Dict]]  # List of papers from Semantic Scholar
    paper_count: int  # Number of papers retrieved
    formatted_results: Optional[str]  # Formatted text response
    json_path: Optional[str]  # Path to saved JSON file
