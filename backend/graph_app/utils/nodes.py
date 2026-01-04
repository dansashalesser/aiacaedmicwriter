"""Node functions for the graph."""

import sys
import asyncio
from pathlib import Path

# Add semantic-scholar folder to path for imports
semantic_scholar_dir = Path(__file__).parent / "semantic-scholar"
sys.path.insert(0, str(semantic_scholar_dir))

from langfuse import observe

from backend.graph_app.utils.state import GraphState
from backend.graph_app.agents.topic_segmenter import segment_topics
from backend.graph_app.agents.knowledge_graph_builder import (
    build_knowledge_graph,
    save_markdown_report
)
from backend.graph_app.agents.paper_proposal_generator import (
    generate_paper_proposals,
    save_proposals_markdown
)
from semantic_scholar import search_papers_async, save_categorized_results, format_categorized_results


@observe(name="topic-segmenter-node")
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


@observe(name="semantic-scholar-node")
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

        # Get filtering parameters from state
        min_citation_count = state.get("min_citation_count")
        publication_types = state.get("publication_types")
        year_range = state.get("year_range")
        open_access_only = state.get("open_access_only", False)
        include_embeddings = state.get("include_embeddings", False)

        print(f"\n🔍 Searching Semantic Scholar:")
        print(f"   • Original query: {user_input[:80]}{'...' if len(user_input) > 80 else ''} ({papers_for_original_query} papers)")
        print(f"   • Individual topics ({len(topics)}): {papers_per_topic} papers each")
        for i, topic in enumerate(topics, 1):
            print(f"      {i}. {topic}")

        # Show active filters
        filters = []
        if min_citation_count:
            filters.append(f"min citations: {min_citation_count}")
        if publication_types:
            filters.append(f"types: {', '.join(publication_types)}")
        if year_range:
            filters.append(f"years: {year_range}")
        if open_access_only:
            filters.append("open access only")
        if include_embeddings:
            filters.append("with embeddings")
        if filters:
            print(f"   • Filters: {', '.join(filters)}")

        # Create parallel search tasks: original query + all topics
        search_tasks = [
            search_papers_async(
                user_input,
                limit=papers_for_original_query,
                min_citation_count=min_citation_count,
                publication_types=publication_types,
                year_range=year_range,
                open_access_only=open_access_only,
                include_embeddings=include_embeddings
            )  # Original query first
        ] + [
            search_papers_async(
                topic,
                limit=papers_per_topic,
                min_citation_count=min_citation_count,
                publication_types=publication_types,
                year_range=year_range,
                open_access_only=open_access_only,
                include_embeddings=include_embeddings
            )
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

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                print(f"   ✗ '{search_query[:50]}...': Invalid result type")
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
            print("WARNING: All searches failed!")
        elif failed_searches:
            print(f"Partial failure: {len(failed_searches)}/{len(all_searches)} searches failed")

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


@observe(name="knowledge-graph-node")
async def knowledge_graph_node(state: GraphState) -> GraphState:
    """
    Build knowledge graph from semantic scholar results.
    Analyzes papers to identify research gaps, contradictions, and connections.
    """
    print("\nBuilding Knowledge Graph...")

    # Extract required state
    papers_by_topic = state.get("papers_by_topic", {})
    topics = state.get("topics", [])
    user_input = state.get("user_input", state.get("query", ""))

    # Validate we have data to analyze
    if not papers_by_topic:
        print("No papers available for analysis")
        return {
            **state,
            "knowledge_graph": None,
            "gap_analysis": {"error": "No papers available"},
            "graph_markdown_path": None
        }

    if not topics:
        print("No topics available for analysis")
        return {
            **state,
            "knowledge_graph": None,
            "gap_analysis": {"error": "No topics available"},
            "graph_markdown_path": None
        }

    total_papers = sum(len(papers) for papers in papers_by_topic.values())
    print(f"   • Analyzing {len(papers_by_topic)} clusters")
    print(f"   • Total papers: {total_papers}")

    try:
        # Build knowledge graph with timeout
        print("   • Running cluster analysis...")
        graph_data = await asyncio.wait_for(
            build_knowledge_graph(papers_by_topic, topics, user_input),
            timeout=300.0  # 5 minute timeout for entire analysis
        )

        # Check if analysis failed
        if "error" in graph_data:
            print(f"   ✗ Analysis failed: {graph_data['error']}")
            return {
                **state,
                "knowledge_graph": graph_data,
                "gap_analysis": {"error": graph_data["error"]},
                "graph_markdown_path": None
            }

        # Save markdown report
        print("   • Generating markdown report...")
        markdown_path = save_markdown_report(graph_data, user_input)

        # Extract gap analysis for easy access
        gap_analysis = graph_data.get("gap_analysis", {})

        print(f"   ✓ Knowledge graph built successfully")
        print(f"   ✓ Report saved: {markdown_path}")

        return {
            **state,
            "knowledge_graph": graph_data,
            "gap_analysis": gap_analysis,
            "graph_markdown_path": markdown_path
        }

    except asyncio.TimeoutError:
        print("   ✗ Timeout: Knowledge graph analysis took too long (>5 minutes)")
        return {
            **state,
            "knowledge_graph": None,
            "gap_analysis": {"error": "Analysis timeout"},
            "graph_markdown_path": None
        }
    except Exception as e:
        print(f"   ✗ Error in knowledge graph node: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "knowledge_graph": None,
            "gap_analysis": {"error": str(e)},
            "graph_markdown_path": None
        }


@observe(name="paper-proposal-node")
async def paper_proposal_node(state: GraphState) -> GraphState:
    """
    Generate detailed paper proposals based on gap analysis.

    Each proposal triggers its own Semantic Scholar search for targeted literature.
    """
    print("\nGenerating Paper Proposals...")

    # Extract required state
    gap_analysis = state.get("gap_analysis", {})
    knowledge_graph = state.get("knowledge_graph", {})
    user_input = state.get("user_input", state.get("query", ""))
    num_proposals = state.get("num_paper_proposals", 5)  # Configurable

    # Validate we have gap analysis
    if not gap_analysis or "error" in gap_analysis:
        print("   No gap analysis available")
        return {
            **state,
            "paper_proposals": None,
            "proposals_markdown_path": None
        }

    cluster_analyses = knowledge_graph.get("cluster_analyses", [])

    try:
        print(f"   • Generating {num_proposals} paper proposals with fresh literature searches...")
        proposals_data = await asyncio.wait_for(
            generate_paper_proposals(gap_analysis, cluster_analyses, user_input, num_proposals),
            timeout=900.0  # 15 minute timeout (allows time for all 5 proposals + journal searches)
        )

        print("   • Saving proposals to markdown...")
        markdown_path = save_proposals_markdown(proposals_data, user_input)

        print(f"   ✓ Generated {len(proposals_data['proposals'])} paper proposals")
        print(f"   ✓ Saved to: {markdown_path}")

        return {
            **state,
            "paper_proposals": proposals_data,
            "proposals_markdown_path": markdown_path
        }

    except asyncio.TimeoutError:
        print("   ✗ Timeout generating proposals - but may have partial results")
        # Note: Partial results are lost on timeout - this is a known limitation
        return {**state, "paper_proposals": None, "proposals_markdown_path": None}
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {**state, "paper_proposals": None, "proposals_markdown_path": None}
