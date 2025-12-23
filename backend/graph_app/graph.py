"""
LangGraph for academic paper research using Semantic Scholar.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langfuse import observe, get_client
from langgraph.graph import StateGraph, END
from backend.graph_app.utils.state import GraphState
from backend.graph_app.utils.nodes import topic_segmenter_node, semantic_scholar_node, knowledge_graph_node, paper_proposal_node


def create_graph():
    """Create and return the graph with topic segmentation, multi-topic search, knowledge graph analysis, and paper proposals."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("topic_segmenter", topic_segmenter_node)
    workflow.add_node("semantic_scholar", semantic_scholar_node)  # Async, handles multiple topics
    workflow.add_node("knowledge_graph_builder", knowledge_graph_node)  # Analyzes papers for gaps
    workflow.add_node("paper_proposal_generator", paper_proposal_node)  # Generates paper proposals

    # Set entry point to topic segmenter
    workflow.set_entry_point("topic_segmenter")

    # Flow: topic_segmenter → semantic_scholar → knowledge_graph_builder → paper_proposal_generator → END
    workflow.add_edge("topic_segmenter", "semantic_scholar")
    workflow.add_edge("semantic_scholar", "knowledge_graph_builder")
    workflow.add_edge("knowledge_graph_builder", "paper_proposal_generator")
    workflow.add_edge("paper_proposal_generator", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


@observe(name="academic-research-pipeline")
async def main():
    """Main async function to run the graph."""
    # Create the graph
    graph = create_graph()

    # Run the graph with initial state
    initial_state = {
        "user_input": "Natural Disaster prediction using LLMs",
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
        "json_path": None,
        # Knowledge graph fields
        "knowledge_graph": None,
        "gap_analysis": None,
        "graph_markdown_path": None,
        # Paper proposal fields
        "num_paper_proposals": 5,
        "paper_proposals": None,
        "proposals_markdown_path": None,
        # Filtering parameters (optional)
        "min_citation_count": None,
        "publication_types": None,
        "year_range": None,
        "open_access_only": False,
        "include_embeddings": False
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
    print(f"Papers JSON: {result['json_path']}")

    # Knowledge graph results
    if result.get('graph_markdown_path'):
        print(f"\n📊 Knowledge Graph Analysis:")
        print(f"   Report: {result['graph_markdown_path']}")

        # Show brief summary if available
        if result.get('knowledge_graph', {}).get('summary'):
            print(f"\n   Summary: {result['knowledge_graph']['summary']}")

        # Show top recommendations if available
        gap_analysis = result.get('gap_analysis', {})
        if gap_analysis and gap_analysis.get('priority_recommendations'):
            print(f"\n   Top Recommendations:")
            for i, rec in enumerate(gap_analysis['priority_recommendations'][:3], 1):
                print(f"   {i}. {rec[:100]}{'...' if len(rec) > 100 else ''}")
    else:
        print(f"\n⚠️  Knowledge graph analysis: {result.get('gap_analysis', {}).get('error', 'Failed')}")

    # Paper proposal results
    if result.get('proposals_markdown_path'):
        print(f"\n📝 Paper Proposals:")
        print(f"   Report: {result['proposals_markdown_path']}")

        # Show proposal titles if available
        if result.get('paper_proposals', {}).get('proposals'):
            proposals = result['paper_proposals']['proposals']
            print(f"\n   Generated {len(proposals)} proposals:")
            for i, proposal in enumerate(proposals[:5], 1):
                print(f"   {i}. {proposal.working_title}")
    else:
        print(f"\n⚠️  Paper proposals: {result.get('paper_proposals', 'Not generated')}")

    print("\n" + result['formatted_results'])

    # Flush langfuse traces before exit
    get_client().flush()

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
