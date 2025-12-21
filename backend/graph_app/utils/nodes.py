"""Node functions for the graph."""

import sys
import asyncio
from pathlib import Path

# Add semantic-scholar folder to path for imports
semantic_scholar_dir = Path(__file__).parent / "semantic-scholar"
sys.path.insert(0, str(semantic_scholar_dir))

from backend.graph_app.utils.state import GraphState
from backend.graph_app.agents.topic_segmenter import segment_topics
from semantic_scholar import search_papers_async, save_categorized_results, format_categorized_results


def topic_segmenter_node(state: GraphState) -> GraphState:
    """
    Entry node: Use LLM to segment user's paper idea into research topics.

    Input: user_input (detailed paper idea)
    Output: topics (list of 1-5 research topics)
    """
    user_input = state["user_input"]
    max_topics = state.get("max_topics", 5)

    print(f"\n🤖 Analyzing paper idea with LLM...")
    print(f"   Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")

    try:
        # Use topic segmentation logic from separate module
        topics = segment_topics(user_input, max_topics)

        print(f"   ✓ Extracted {len(topics)} topics:")
        for i, topic in enumerate(topics, 1):
            print(f"      {i}. {topic}")

    except Exception as e:
        print(f"   ⚠️  LLM segmentation failed: {e}")
        print(f"   Using fallback: entire input as single topic")
        topics = [user_input]

    return {
        **state,  # Preserve all existing state
        "topics": topics,
        "query": user_input  # For backwards compatibility
    }


async def semantic_scholar_node(state: GraphState) -> GraphState:
    """
    Search Semantic Scholar API for papers based on topics.
    Handles both new flow (multiple topics in parallel) and old flow (single query).
    """
    # Check if we have topics from the new flow
    topics = state.get("topics")

    if topics and len(topics) > 0:
        # NEW FLOW: Search original query + individual topics in parallel
        user_input = state.get("user_input", state.get("query", ""))
        papers_per_topic = state.get("papers_per_topic", 5)
        papers_for_original_query = state.get("papers_for_original_query", 10)

        print(f"\n🔍 Searching Semantic Scholar:")
        print(f"   • Original query: {user_input[:80]}{'...' if len(user_input) > 80 else ''} ({papers_for_original_query} papers)")
        print(f"   • Individual topics ({len(topics)}): {papers_per_topic} papers each")
        for i, topic in enumerate(topics, 1):
            print(f"      {i}. {topic}")

        # Create parallel search tasks: original query + all topics
        search_tasks = [
            search_papers_async(user_input, limit=papers_for_original_query)  # Original query first
        ] + [
            search_papers_async(topic, limit=papers_per_topic)
            for topic in topics
        ]

        # Track which search is for original query vs topics
        all_searches = [user_input] + topics

        # Execute in parallel with timeout protection
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            print("⚠️  Search timeout - using partial results")
            results = [{'total': 0, 'data': [], 'error': 'Timeout'} for _ in topics]

        # Process results and categorize by search query/topic
        papers_by_topic = {}
        total_papers = 0
        failed_searches = []

        for search_query, result in zip(all_searches, results):
            # Handle exceptions from gather
            if isinstance(result, Exception):
                print(f"   ✗ '{search_query[:50]}...': {result}")
                papers_by_topic[search_query] = []
                failed_searches.append(search_query)
                continue

            # Handle error results
            if 'error' in result:
                print(f"   ✗ '{search_query[:50]}...': {result['error']}")
                papers_by_topic[search_query] = []
                failed_searches.append(search_query)
                continue

            # Success case - tag each paper with its category
            papers = result.get('data', [])
            tagged_papers = []
            for paper in papers:
                paper_copy = paper.copy()
                paper_copy['category'] = search_query
                tagged_papers.append(paper_copy)

            papers_by_topic[search_query] = tagged_papers
            total_papers += len(tagged_papers)

            # Show different label for original query vs topic
            label = "Original Query" if search_query == user_input else f"'{search_query}'"
            print(f"   ✓ {label}: {len(papers)} papers (of {result.get('total', 0)} total)")

        # Warning if all searches failed
        if len(failed_searches) == len(all_searches):
            print("⚠️  WARNING: All searches failed!")
        elif failed_searches:
            print(f"⚠️  Partial failure: {len(failed_searches)}/{len(all_searches)} searches failed")

        # Save consolidated results to JSON (original query + topics)
        json_path = save_categorized_results(user_input, papers_by_topic, all_searches)

        # Format results for display
        formatted = format_categorized_results(papers_by_topic, user_input)

        return {
            **state,
            "papers_by_topic": papers_by_topic,
            "total_papers": total_papers,
            "paper_count": total_papers,  # Backwards compatibility
            "formatted_results": formatted,
            "json_path": json_path
        }

    else:
        # OLD FLOW: Fallback to single query (not expected in new implementation)
        print("⚠️  Warning: No topics found, this shouldn't happen in the new flow")
        return {
            **state,
            "papers_by_topic": {},
            "total_papers": 0,
            "paper_count": 0,
            "formatted_results": "No topics to search",
            "json_path": None
        }
