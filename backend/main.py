"""Main entry point for the Academic Paper Research Agent."""

import sys
import asyncio
from pathlib import Path

# Add the graph_app directory to the Python path
graph_app_path = Path(__file__).parent / "graph_app"
sys.path.insert(0, str(graph_app_path))

from graph_app.graph import create_graph
from langfuse import get_client


async def main():
    """Run the complete academic paper research pipeline."""
    print("=" * 80)
    print("ACADEMIC PAPER RESEARCH AGENT - STRENGTHENED PROPOSAL GENERATION")
    print("=" * 80)
    print("\nThis system will:")
    print("  1. Segment your research interest into topics")
    print("  2. Search Semantic Scholar for relevant papers")
    print("  3. Build a knowledge graph and identify intellectual gaps")
    print("  4. Generate validated, distinct research proposals")
    print("\n" + "=" * 80)

    # ===========================================================================
    # CONFIGURATION - Edit this section to customize your research
    # ===========================================================================

    # YOUR RESEARCH INTEREST - Edit this placeholder to your actual research topic
    research_query = "Creating an LLM based algo-trading system that can trade on the stock market with a high accuracy rate."

    # OPTIONAL: Customize these parameters
    num_proposals = 5          # Number of proposals to generate (default: 5)
    max_topics = 5            # Maximum topics to segment into (default: 5)
    papers_per_topic = 20      # Papers to fetch per topic (default: 5)
    papers_for_query = 10     # Papers for original query (default: 10)

    # OPTIONAL: Add filters (uncomment and adjust as needed)
    min_citation_count = 30   # Minimum citations required
    # year_range = (2019, 2024) # Only papers from these years
    # open_access_only = True   # Only open access papers

    # ===========================================================================

    print(f"\n📚 Research Interest: {research_query}")
    print(f"📊 Will generate: {num_proposals} proposals")
    print(f"🔍 Search strategy: {max_topics} topics, {papers_per_topic} papers/topic + {papers_for_query} for main query")
    print("\n" + "=" * 80 + "\n")

    # Create the graph
    graph = create_graph()

    # Build initial state
    initial_state = {
        "user_input": research_query,
        "topics": None,
        "max_topics": max_topics,
        "papers_per_topic": papers_per_topic,
        "papers_for_original_query": papers_for_query,
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
        "num_paper_proposals": num_proposals,
        "paper_proposals": None,
        "proposals_markdown_path": None,
        # Optional filters (uncomment if defined above)
        "min_citation_count": None,  # or min_citation_count if defined
        "publication_types": None,
        "year_range": None,          # or year_range if defined
        "open_access_only": False,   # or open_access_only if defined
        "include_embeddings": False
    }

    # Run the graph
    try:
        result = await graph.ainvoke(initial_state)

        # Display results
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE - RESULTS SUMMARY")
        print("=" * 80)

        print(f"\n📚 Original Research Interest:")
        print(f"   {result['user_input']}")

        print(f"\n🔍 Topics Searched:")
        if result.get('topics'):
            for i, topic in enumerate(result['topics'], 1):
                print(f"   {i}. {topic}")

        print(f"\n📄 Literature Collection:")
        print(f"   Total Papers: {result.get('total_papers', 0)}")
        if result.get('json_path'):
            print(f"   JSON Export: {result['json_path']}")

        # Knowledge graph results
        print(f"\n📊 Knowledge Graph Analysis:")
        if result.get('graph_markdown_path'):
            print(f"   ✓ Report saved: {result['graph_markdown_path']}")

            # Show gap quality score if available
            gap_analysis = result.get('gap_analysis', {})
            if gap_analysis and 'gap_quality_score' in gap_analysis:
                score = gap_analysis['gap_quality_score']
                quality = "Excellent (intellectual)" if score > 0.7 else "Good" if score > 0.5 else "Needs improvement"
                print(f"   ✓ Gap Quality Score: {score:.2f} ({quality})")

            # Show intellectual gaps
            if gap_analysis and gap_analysis.get('intellectual_gaps'):
                print(f"\n   Intellectual Gaps Identified ({len(gap_analysis['intellectual_gaps'])}):")
                for i, gap in enumerate(gap_analysis['intellectual_gaps'][:3], 1):
                    print(f"   {i}. {gap[:120]}{'...' if len(gap) > 120 else ''}")

            # Show priority recommendations
            if gap_analysis and gap_analysis.get('priority_recommendations'):
                print(f"\n   Top Recommendations:")
                for i, rec in enumerate(gap_analysis['priority_recommendations'][:3], 1):
                    print(f"   {i}. {rec[:120]}{'...' if len(rec) > 120 else ''}")
        else:
            print(f"   ✗ Analysis failed: {result.get('gap_analysis', {}).get('error', 'Unknown error')}")

        # Paper proposal results
        print(f"\n📝 Research Proposals:")
        if result.get('proposals_markdown_path'):
            print(f"   ✓ Report saved: {result['proposals_markdown_path']}")

            # Show proposal details
            paper_proposals = result.get('paper_proposals', {})
            proposals = paper_proposals.get('proposals', [])
            rejected = paper_proposals.get('rejected_proposals', [])
            requested = result.get('num_paper_proposals', 5)

            # Show generation status
            if len(proposals) == requested:
                print(f"   ✓ Generated: {len(proposals)}/{requested} valid proposals")
            elif len(proposals) > 0:
                print(f"   ⚠️  Partial success: {len(proposals)}/{requested} proposals generated")
            else:
                print(f"   ✗ No valid proposals generated (0/{requested})")

            if rejected:
                print(f"   ⚠️  Rejected: {len(rejected)} proposals (failed validation)")

            # Show average similarity if available
            # Note: This would need to be added to the return value if you want to display it

            if proposals:
                print(f"\n   Proposal Titles:")
                for i, proposal in enumerate(proposals, 1):
                    print(f"   {i}. {proposal.working_title}")

            # Show validation statistics if rejected proposals exist
            if rejected:
                print(f"\n   Rejected Proposals (for debugging):")
                for i, reject_info in enumerate(rejected[:3], 1):
                    print(f"   {i}. {reject_info['concept']}")
                    if reject_info.get('errors'):
                        print(f"      Reason: {reject_info['errors'][0][:100]}...")
        else:
            print(f"   ✗ Proposal generation failed completely (no proposals generated)")

        print("\n" + "=" * 80)
        print("📁 All results saved to markdown reports (see paths above)")
        print("=" * 80 + "\n")

        # Flush langfuse traces
        get_client().flush()

        return result

    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with exception:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()

        # Flush langfuse even on error
        get_client().flush()

        return None


if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if result else 1)
