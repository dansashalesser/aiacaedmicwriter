"""
LangGraph for academic paper research using Semantic Scholar.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, END
from backend.graph_app.utils.state import GraphState
from backend.graph_app.utils.nodes import semantic_scholar_node


def create_graph():
    """Create and return the graph."""
    workflow = StateGraph(GraphState)

    # Add the semantic scholar search node
    workflow.add_node("semantic_scholar", semantic_scholar_node)

    # Set entry point to semantic scholar
    workflow.set_entry_point("semantic_scholar")

    # Add edge from semantic scholar to END (will add more nodes later)
    workflow.add_edge("semantic_scholar", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


if __name__ == "__main__":
    # Create the graph
    graph = create_graph()

    # Run the graph with initial state
    initial_state = {
        "query": "large language models",
        "papers": None,
        "paper_count": 0,
        "formatted_results": None,
        "json_path": None
    }

    print("=" * 60)
    print("ACADEMIC PAPER RESEARCH AGENT")
    print("=" * 60)

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nQuery: {result['query']}")
    print(f"Papers Retrieved: {result['paper_count']}")
    print("\n" + result['formatted_results'])
