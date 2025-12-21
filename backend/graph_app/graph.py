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
from backend.graph_app.utils.nodes import topic_segmenter_node, semantic_scholar_node


def create_graph():
    """Create and return the graph with topic segmentation and multi-topic search."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("topic_segmenter", topic_segmenter_node)
    workflow.add_node("semantic_scholar", semantic_scholar_node)  # Now async, handles multiple topics

    # Set entry point to topic segmenter
    workflow.set_entry_point("topic_segmenter")

    # NEW flow: topic_segmenter → semantic_scholar → END
    workflow.add_edge("topic_segmenter", "semantic_scholar")
    workflow.add_edge("semantic_scholar", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


async def main():
    """Main async function to run the graph."""
    # Create the graph
    graph = create_graph()

    # Run the graph with initial state
    initial_state = {
        "user_input": "LLM for environmental prediction",
        "topics": None,
        "max_topics": 5,
        "papers_per_topic": 5,  # Papers per individual topic
        "papers_for_original_query": 10,  # Papers for the original query
        "papers_by_topic": None,
        "total_papers": 0,
        "query": "",
        "papers": None,
        "paper_count": 0,
        "formatted_results": None,
        "json_path": None
    }

    print("=" * 60)
    print("ACADEMIC PAPER RESEARCH AGENT - MULTI-TOPIC SEARCH")
    print("=" * 60)

    # Use ainvoke for async execution
    result = await graph.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOriginal Idea: {result['user_input']}")
    print(f"Topics Searched: {', '.join(result['topics'])}")
    print(f"Total Papers: {result['total_papers']}")
    print(f"JSON saved to: {result['json_path']}")
    print("\n" + result['formatted_results'])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
