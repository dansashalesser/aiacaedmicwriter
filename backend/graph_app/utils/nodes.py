"""Node functions for the graph."""

import sys
from pathlib import Path

# Add semantic-scholar folder to path for imports
semantic_scholar_dir = Path(__file__).parent / "semantic-scholar"
sys.path.insert(0, str(semantic_scholar_dir))

from backend.graph_app.utils.state import GraphState
from semantic_scholar import search_papers, format_paper_results


def semantic_scholar_node(state: GraphState) -> GraphState:
    """
    First node: Search Semantic Scholar API for papers based on query.
    Retrieves papers, sorts by citations, saves to JSON, and formats results.
    """
    query = state["query"]
    print(f"\n🔍 Searching Semantic Scholar for: '{query}'")

    # Search papers (default: 50 papers, sorted by citations, saved to JSON)
    results = search_papers(query, limit=50)

    # Extract paper data
    papers = results.get("data", [])
    total_found = results.get("total", 0)

    # Format results for display
    formatted = format_paper_results(results)

    print(f"✓ Found {len(papers)} papers (out of {total_found} total)")

    return {
        "query": query,
        "papers": papers,
        "paper_count": len(papers),
        "formatted_results": formatted,
        "json_path": None  # Path is printed by search_papers but not returned
    }
