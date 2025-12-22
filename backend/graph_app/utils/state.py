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

    # NEW: Advanced filtering parameters
    min_citation_count: Optional[int]  # Minimum citation threshold (e.g., 10)
    publication_types: Optional[List[str]]  # Filter by ['Review', 'JournalArticle', 'Conference']
    year_range: Optional[str]  # Year or year range (e.g., '2023' or '2020-2024')
    open_access_only: bool  # Only return papers with public PDFs (default: False)
    include_embeddings: bool  # Include specter_v2 embeddings for semantic similarity (default: False)

    # NEW: Categorized results
    papers_by_topic: Optional[Dict[str, List[Dict]]]  # Papers grouped by topic (includes original query)
    total_papers: int  # Sum across original query + all topics

    # EXISTING: Keep for backwards compatibility
    query: str  # User's search query
    papers: Optional[List[Dict]]  # List of papers from Semantic Scholar
    paper_count: int  # Number of papers retrieved
    formatted_results: Optional[str]  # Formatted text response
    json_path: Optional[str]  # Path to saved JSON file

    # NEW: Knowledge graph analysis
    knowledge_graph: Optional[Dict]  # Full knowledge graph analysis output
    gap_analysis: Optional[Dict]  # Research gaps and untried directions
    graph_markdown_path: Optional[str]  # Path to saved markdown report
