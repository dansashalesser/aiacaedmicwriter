"""Knowledge Graph Builder - Analyzes research papers to identify gaps and connections."""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime
from pathlib import Path
import re
import os
from dotenv import load_dotenv
from langfuse import observe

# Load environment variables
load_dotenv()

# ==============================================================================
# Pydantic Models for Structured Outputs
# ==============================================================================

class MethodAnalysis(BaseModel):
    """Analysis of a common method/approach used across papers."""
    method: str = Field(description="Name of the method/approach")
    papers_using: List[str] = Field(description="Paper titles using this method")
    frequency: int = Field(description="Number of papers using this method")
    citation_weight: float = Field(description="Weighted importance based on citations")


class LimitationAnalysis(BaseModel):
    """Analysis of a limitation mentioned in papers."""
    limitation: str = Field(description="Description of the limitation")
    mentioned_in: List[str] = Field(description="Paper titles mentioning this limitation")
    severity: str = Field(description="Severity level: minor, moderate, major")
    potential_solution: Optional[str] = Field(description="Possible way to address this limitation")


class ContradictionAnalysis(BaseModel):
    """Analysis of contradictory findings between papers."""
    finding_a: str = Field(description="First contradictory finding")
    finding_b: str = Field(description="Second contradictory finding")
    papers_a: List[str] = Field(description="Papers supporting finding A")
    papers_b: List[str] = Field(description="Papers supporting finding B")
    explanation: str = Field(description="Why this contradiction matters")


class MissingComparison(BaseModel):
    """Identification of missing method comparisons."""
    method_a: str = Field(description="First method that should be compared")
    method_b: str = Field(description="Second method that should be compared")
    reason: str = Field(description="Why this comparison would be valuable")


class ClusterAnalysis(BaseModel):
    """Complete analysis of a single research cluster/topic."""
    cluster_name: str
    paper_count: int
    papers_with_abstracts: int = Field(description="Number of papers with available abstracts")
    common_methods: List[MethodAnalysis]
    limitations: List[LimitationAnalysis]
    contradictions: List[ContradictionAnalysis]
    missing_comparisons: List[MissingComparison]
    untested_assumptions: List[str] = Field(description="Assumptions papers make without testing", default_factory=list)
    measurement_gaps: List[str] = Field(description="Concepts lacking proper operationalization", default_factory=list)
    key_insights: str = Field(description="High-level summary of this cluster")


class GapAnalysis(BaseModel):
    """Overall gap analysis across all clusters."""
    intellectual_gaps: List[str] = Field(description="Untested assumptions and unresolved contradictions in the literature")
    methodological_gaps: List[str] = Field(description="Measurement and design limitations that prevent answering key questions")
    research_gaps: List[str] = Field(description="DEPRECATED: Use intellectual_gaps instead", default_factory=list)
    untried_directions: List[str] = Field(description="DEPRECATED: Specific research directions", default_factory=list)
    priority_recommendations: List[str] = Field(description="Top 3-5 most promising directions with rationale")
    gap_quality_score: float = Field(description="Self-assessment of gap quality (0-1, where 1=intellectual, 0=coverage)", default=0.5)


class KnowledgeGraphOutput(BaseModel):
    """Complete knowledge graph analysis output."""
    cluster_analyses: List[ClusterAnalysis]
    gap_analysis: GapAnalysis
    summary: str = Field(description="Executive summary of key findings")


# ==============================================================================
# Helper Functions
# ==============================================================================

@observe(name="calculate-citation-weight")
def calculate_citation_weight(papers: List[Dict]) -> List[Dict]:
    """
    Normalize citation counts and add weight field to each paper.
    Weight ranges from 0.1 (low citations) to 1.0 (highest citations).
    """
    if not papers:
        return papers

    max_citations = max(p.get('citationCount', 0) for p in papers)
    if max_citations == 0:
        max_citations = 1

    for paper in papers:
        citations = paper.get('citationCount', 0)
        # Weight from 0.1 to 1.0
        paper['citation_weight'] = 0.1 + (0.9 * citations / max_citations)

    # Sort by citation weight (highest first)
    return sorted(papers, key=lambda p: p.get('citation_weight', 0), reverse=True)


@observe(name="filter-papers-with-abstracts")
def filter_papers_with_abstracts(papers: List[Dict]) -> tuple[List[Dict], int]:
    """
    Filter papers that have abstracts.
    Returns (papers_with_abstracts, excluded_count).
    """
    papers_with_abstracts = [p for p in papers if p.get('abstract')]
    excluded_count = len(papers) - len(papers_with_abstracts)
    return papers_with_abstracts, excluded_count


@observe(name="sanitize-filename")
def sanitize_filename(user_input: str) -> str:
    """Sanitize user input to create a valid filename."""
    # Take first 50 chars, replace non-alphanumeric with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', user_input[:50])
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Trim underscores from ends
    return sanitized.strip('_')


@observe(name="format-papers-for-prompt")
def format_papers_for_prompt(papers: List[Dict]) -> str:
    """Format papers for LLM prompt."""
    formatted = []
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Unknown')
        abstract = paper.get('abstract', 'No abstract available')
        year = paper.get('year', 'N/A')
        citations = paper.get('citationCount', 0)
        weight = paper.get('citation_weight', 0)

        formatted.append(
            f"{i}. **{title}** ({year})\n"
            f"   - Citations: {citations} (weight: {weight:.2f})\n"
            f"   - Abstract: {abstract[:500]}...\n"
        )

    return "\n".join(formatted)


# ==============================================================================
# Cluster Analysis
# ==============================================================================

@observe(name="analyze-cluster", as_type="generation")
async def analyze_cluster(
    cluster_name: str,
    papers: List[Dict],
    llm: ChatOpenAI
) -> Optional[ClusterAnalysis]:
    """
    Analyze a single cluster of papers.
    Returns ClusterAnalysis or None if analysis fails.
    """
    try:
        # Filter and weight papers
        papers_weighted = calculate_citation_weight(papers)
        papers_with_abstracts, excluded_count = filter_papers_with_abstracts(papers_weighted)

        if not papers_with_abstracts:
            print(f"Warning: No papers with abstracts in cluster '{cluster_name}'")
            return None

        # Format papers for prompt
        papers_text = format_papers_for_prompt(papers_with_abstracts)

        # Create structured LLM
        structured_llm = llm.with_structured_output(ClusterAnalysis)

        # Cluster analysis prompt
        prompt = f"""You are analyzing academic papers in the research area: "{cluster_name}"

Papers provided (sorted by citation count, higher citations = more established):

{papers_text}

Tasks:
1. Identify common methods/approaches used across papers. Methods should be SCIENTIFIC and SPECIFIC to the cluster, and NOT GENERALIZED methods.
   - Weight by citations: highly-cited papers indicate established methods
   - List paper titles using each method
   - Calculate frequency (number of papers using it)
   - Estimate citation weight (average weight of papers using this method)
   - Provide the ACTUAL 

2. Extract explicit limitations mentioned by authors
   - Extract ACTUAL specific limitations, NOT generalized concepts (for instance, 'Lack of high-resolution mapping data for rocky terrains', instead of 'Missing environmental variables')
   - Classify severity: minor, moderate, major
   - Suggest potential solutions where applicable. Solutions should be SPECIFIC and not GENERALIZED
   - List which papers mention each limitation

3. Detect contradictory findings between papers, if any
   - There could be NO CONTRADICTIONS, so ONLY DETECT these if they exist! **THIS IS CRUCIAL!**
   - Explain why the contradiction matters, and BE SPECIFIC about what is contradicted
   - List papers supporting each side
   - Focus on significant contradictions, not minor differences

4. Note missing method comparisons
   - There could be NO MISSING METHODS, so ONLY DETECT THESE if if they exist! **THIS IS CRUCIAL!**
   - Identify methods that should be compared but aren't
   - Explain why the comparison would be valuable

5. Extract UNTESTED ASSUMPTIONS from papers
   - What do authors ASSUME without empirical testing? (e.g., "assumes users process information rationally", "assumes linear relationship between X and Y")
   - What SCOPE LIMITATIONS do they acknowledge? (e.g., "limited to coastal regions", "excludes rural populations")
   - What DATA LIMITATIONS create inference gaps? (e.g., "lacks individual-level behavior data", "relies on self-reported measures")
   - Be SPECIFIC: quote paper titles and mention the actual assumption
   - If no clear assumptions found, return empty list

6. Identify MEASUREMENT GAPS
   - Which concepts lack proper operationalization? (e.g., "panic measured only via survey, not actual behavior")
   - Which variables use poor proxies? (e.g., "evacuation intent used as proxy for evacuation action")
   - Which constructs are mentioned but not measured? (e.g., "discusses 'trust' but never operationalizes it")
   - If no measurement gaps found, return empty list

7. Provide key insights summary for this cluster (2-3 sentences)

Focus on concrete, actionable findings. Be specific and cite paper titles.

IMPORTANT: Set cluster_name to "{cluster_name}", paper_count to {len(papers)}, and papers_with_abstracts to {len(papers_with_abstracts)}."""

        # Invoke LLM with timeout
        result = await asyncio.wait_for(
            structured_llm.ainvoke(prompt),
            timeout=60.0
        )

        # Ensure result is ClusterAnalysis type
        if isinstance(result, dict):
            result = ClusterAnalysis(**result)

        print(f"✓ Analyzed cluster: {cluster_name} ({len(papers_with_abstracts)} papers)")
        return result

    except asyncio.TimeoutError:
        print(f"✗ Timeout analyzing cluster: {cluster_name}")
        return None
    except Exception as e:
        print(f"✗ Error analyzing cluster '{cluster_name}': {e}")
        return None


# ==============================================================================
# Gap Analysis Synthesis
# ==============================================================================

@observe(name="synthesize-gap-analysis", as_type="generation")
async def synthesize_gap_analysis(
    cluster_analyses: List[ClusterAnalysis],
    user_input: str,
    llm: ChatOpenAI
) -> Optional[GapAnalysis]:
    """
    Synthesize gap analysis from all cluster analyses.
    Returns GapAnalysis or None if synthesis fails.
    """
    try:
        # Format cluster summaries
        cluster_summaries = []
        for cluster in cluster_analyses:
            summary = (
                f"**{cluster.cluster_name}** ({cluster.paper_count} papers)\n"
                f"- Key methods: {', '.join([m.method for m in cluster.common_methods[:3]])}\n"
                f"- Main limitations: {', '.join([l.limitation[:50] for l in cluster.limitations[:2]])}\n"
                f"- Untested assumptions: {', '.join([a[:60] for a in cluster.untested_assumptions[:2]]) if cluster.untested_assumptions else 'None identified'}\n"
                f"- Measurement gaps: {', '.join([m[:60] for m in cluster.measurement_gaps[:2]]) if cluster.measurement_gaps else 'None identified'}\n"
                f"- Insights: {cluster.key_insights}\n"
            )
            cluster_summaries.append(summary)

        clusters_text = "\n".join(cluster_summaries)

        # Create structured LLM
        structured_llm = llm.with_structured_output(GapAnalysis)

        # Gap synthesis prompt
        prompt = f"""You are a research strategist identifying INTELLECTUAL GAPS in the literature.

Cluster analyses provided:

{clusters_text}

Original research question: "{user_input}"

**CRITICAL INSTRUCTIONS - READ CAREFULLY**:

Your job is to identify INTELLECTUAL GAPS, not coverage gaps.

❌ **DO NOT** suggest gaps like:
- "No one has combined X and Y"
- "Domain Z has not been studied with method M"
- "There is limited research on [topic]"
- "Few studies have explored [combination]"

✅ **DO** identify gaps like:
- "Existing models ASSUME X, but this assumption breaks down when Y occurs—no studies test whether..."
- "Papers contradict each other on whether A causes B, suggesting we don't understand the boundary conditions..."
- "Authors acknowledge limitation L but don't address it, leaving question Q unanswered..."
- "Construct C is discussed but never properly operationalized, preventing measurement of..."

Tasks:

1. Identify INTELLECTUAL GAPS (3-5 gaps)
   - What ASSUMPTIONS do papers make that remain untested?
   - What CONTRADICTIONS exist that reveal theoretical uncertainty?
   - What LIMITATIONS do authors acknowledge but not resolve?
   - Each gap should explain: (a) what we think we know, (b) why that belief is questionable, (c) what understanding is missing
   - **CRITICAL**: Every gap MUST include the word "assume", "assumption", "unclear", "untested", "contradict", or "don't know"
   - **CRITICAL**: Include specific numbers and quantification where possible

   Example format: "Current models assume [assumption with numbers]. However, [evidence of limitation with citation]. We don't understand [specific knowledge gap], which would distinguish between [theory A predicting X%] and [theory B predicting Y%]."

2. Identify METHODOLOGICAL GAPS (2-3 gaps)
   - Which variables/constructs lack proper measurement approaches?
   - Which causal mechanisms are discussed but never empirically tested?
   - Which design choices lack justification or validation?

   Example format: "[Construct] is measured using [proxy], but this fails to capture [aspect]. No studies validate whether [proxy] actually reflects [construct] in [context]."

3. Self-assess gap quality (gap_quality_score: 0.0 to 1.0)
   - Score each gap: Does it identify an UNTESTED ASSUMPTION/CONTRADICTION (score ~1.0) or just an UNSTUDIED COMBINATION (score ~0.0)?
   - Return average score across all gaps
   - If average < 0.6, revise gaps to be more intellectually substantive

4. Provide 3-5 priority recommendations
   - Each should address an intellectual or methodological gap identified above
   - Include specific rationale: Why does this gap matter? What would resolving it change?
   - Prioritize by: (a) theoretical importance, (b) feasibility, (c) potential impact

Focus on what we DON'T UNDERSTAND, not what we HAVEN'T TRIED."""

        # Invoke LLM with timeout
        result = await asyncio.wait_for(
            structured_llm.ainvoke(prompt),
            timeout=90.0
        )

        # Ensure result is GapAnalysis type
        if isinstance(result, dict):
            result = GapAnalysis(**result)

        print(f"✓ Synthesized gap analysis across {len(cluster_analyses)} clusters")
        return result

    except asyncio.TimeoutError:
        print("✗ Timeout during gap analysis synthesis")
        return None
    except Exception as e:
        print(f"✗ Error during gap analysis synthesis: {e}")
        return None


# ==============================================================================
# Main Knowledge Graph Builder
# ==============================================================================

@observe(name="build-knowledge-graph")
async def build_knowledge_graph(
    papers_by_topic: Dict[str, List[Dict]],
    topics: List[str],
    user_input: str
) -> Dict[str, Any]:
    """
    Main function to analyze papers and build knowledge graph.

    Args:
        papers_by_topic: Dictionary mapping topics to lists of papers
        topics: List of research topics
        user_input: Original research query

    Returns:
        Dictionary with full analysis results
    """
    print("\n=== Building Knowledge Graph ===")
    print(f"Topics to analyze: {len(papers_by_topic)}")
    print(f"Total papers: {sum(len(papers) for papers in papers_by_topic.values())}")

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )

    # Stage 1: Analyze each cluster in parallel
    print("\n[Stage 1] Analyzing clusters in parallel...")
    cluster_tasks = [
        analyze_cluster(topic, papers, llm)
        for topic, papers in papers_by_topic.items()
        if papers  # Skip empty clusters
    ]

    try:
        cluster_results = await asyncio.wait_for(
            asyncio.gather(*cluster_tasks, return_exceptions=True),
            timeout=180.0
        )
    except asyncio.TimeoutError:
        print("✗ Timeout during cluster analysis (180s limit exceeded)")
        cluster_results = []

    # Filter out None results and exceptions
    cluster_analyses = [
        result for result in cluster_results
        if result is not None and isinstance(result, ClusterAnalysis)
    ]

    if not cluster_analyses:
        print("✗ No successful cluster analyses")
        return {
            "error": "Failed to analyze any clusters",
            "cluster_analyses": [],
            "gap_analysis": None,
            "summary": "Analysis failed"
        }

    print(f"✓ Successfully analyzed {len(cluster_analyses)}/{len(cluster_tasks)} clusters")

    # Stage 2: Synthesize gap analysis
    print("\n[Stage 2] Synthesizing gap analysis...")
    gap_analysis = await synthesize_gap_analysis(cluster_analyses, user_input, llm)

    if gap_analysis is None:
        print("✗ Gap analysis synthesis failed")
        gap_analysis = GapAnalysis(
            intellectual_gaps=["Gap analysis failed"],
            methodological_gaps=["Analysis incomplete"],
            priority_recommendations=["Re-run analysis"],
            gap_quality_score=0.0
        )

    # Stage 3: Generate executive summary
    print("\n[Stage 3] Generating executive summary...")
    summary = f"Analyzed {len(cluster_analyses)} research clusters covering {sum(c.paper_count for c in cluster_analyses)} papers. "
    summary += f"Identified {len(gap_analysis.intellectual_gaps)} intellectual gaps and {len(gap_analysis.methodological_gaps)} methodological gaps. "
    summary += f"Gap quality score: {gap_analysis.gap_quality_score:.2f}. "
    summary += f"Top priority: {gap_analysis.priority_recommendations[0] if gap_analysis.priority_recommendations else 'None'}"

    # Build output
    output = KnowledgeGraphOutput(
        cluster_analyses=cluster_analyses,
        gap_analysis=gap_analysis,
        summary=summary
    )

    print("✓ Knowledge graph built successfully\n")

    # Convert to dictionary
    return output.model_dump()


# ==============================================================================
# Markdown Report Generation
# ==============================================================================

@observe(name="save-markdown-report")
def save_markdown_report(output: Dict[str, Any], user_input: str) -> str:
    """
    Generate and save markdown report to results folder.

    Args:
        output: Knowledge graph output dictionary
        user_input: Original research query

    Returns:
        Path to saved markdown file
    """
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate filename
    sanitized_query = sanitize_filename(user_input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_query}_{timestamp}.md"
    filepath = results_dir / filename

    # Extract data
    cluster_analyses = output.get("cluster_analyses", [])
    gap_analysis = output.get("gap_analysis", {})
    summary = output.get("summary", "")

    # Build markdown content
    md_lines = [
        f"# Knowledge Graph Analysis: {user_input}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Clusters Analyzed**: {len(cluster_analyses)}",
        f"**Total Papers**: {sum(c.get('paper_count', 0) for c in cluster_analyses)}",
        "",
        "## Executive Summary",
        "",
        summary,
        "",
        "---",
        "",
        "## Cluster Analysis",
        ""
    ]

    # Add each cluster analysis
    for cluster in cluster_analyses:
        cluster_name = cluster.get("cluster_name", "Unknown")
        paper_count = cluster.get("paper_count", 0)
        papers_with_abstracts = cluster.get("papers_with_abstracts", 0)

        md_lines.extend([
            f"### Cluster: {cluster_name}",
            f"**Papers**: {paper_count} ({papers_with_abstracts} with abstracts)",
            ""
        ])

        # Common methods
        common_methods = cluster.get("common_methods", [])
        if common_methods:
            md_lines.append("#### Common Methods")
            for i, method in enumerate(common_methods, 1):
                method_name = method.get("method", "")
                frequency = method.get("frequency", 0)
                weight = method.get("citation_weight", 0)
                papers = method.get("papers_using", [])
                md_lines.extend([
                    f"{i}. **{method_name}** (used in {frequency} papers, avg citation weight: {weight:.2f})",
                    f"   - Papers: {', '.join(papers[:3])}{'...' if len(papers) > 3 else ''}",
                    ""
                ])

        # Limitations
        limitations = cluster.get("limitations", [])
        if limitations:
            md_lines.append("#### Limitations Identified")
            for limitation in limitations:
                limit_text = limitation.get("limitation", "")
                severity = limitation.get("severity", "")
                mentioned_in = limitation.get("mentioned_in", [])
                solution = limitation.get("potential_solution", "")
                md_lines.extend([
                    f"- **{limit_text}** (Severity: {severity})",
                    f"  - Mentioned in: {', '.join(mentioned_in[:2])}{'...' if len(mentioned_in) > 2 else ''}",
                ])
                if solution:
                    md_lines.append(f"  - Potential solution: {solution}")
                md_lines.append("")

        # Contradictions
        contradictions = cluster.get("contradictions", [])
        if contradictions:
            md_lines.append("#### Contradictions Found")
            for contradiction in contradictions:
                finding_a = contradiction.get("finding_a", "")
                finding_b = contradiction.get("finding_b", "")
                papers_a = contradiction.get("papers_a", [])
                papers_b = contradiction.get("papers_b", [])
                explanation = contradiction.get("explanation", "")
                md_lines.extend([
                    f"- **{finding_a}** vs **{finding_b}**",
                    f"  - Supporting A: {', '.join(papers_a[:2])}",
                    f"  - Supporting B: {', '.join(papers_b[:2])}",
                    f"  - Why it matters: {explanation}",
                    ""
                ])

        # Missing comparisons
        missing_comparisons = cluster.get("missing_comparisons", [])
        if missing_comparisons:
            md_lines.append("#### Missing Comparisons")
            for comparison in missing_comparisons:
                method_a = comparison.get("method_a", "")
                method_b = comparison.get("method_b", "")
                reason = comparison.get("reason", "")
                md_lines.append(f"- {method_a} vs {method_b}: {reason}")
            md_lines.append("")

        # Untested assumptions
        untested_assumptions = cluster.get("untested_assumptions", [])
        if untested_assumptions:
            md_lines.append("#### Untested Assumptions")
            for assumption in untested_assumptions:
                md_lines.append(f"- {assumption}")
            md_lines.append("")

        # Measurement gaps
        measurement_gaps = cluster.get("measurement_gaps", [])
        if measurement_gaps:
            md_lines.append("#### Measurement Gaps")
            for gap in measurement_gaps:
                md_lines.append(f"- {gap}")
            md_lines.append("")

        # Key insights
        key_insights = cluster.get("key_insights", "")
        if key_insights:
            md_lines.extend([
                "#### Key Insights",
                key_insights,
                ""
            ])

        md_lines.extend(["---", ""])

    # Gap analysis
    md_lines.extend([
        "## Gap Analysis",
        "",
        f"**Gap Quality Score:** {gap_analysis.get('gap_quality_score', 0.0):.2f}/1.0 (1.0 = intellectual gaps, 0.0 = coverage gaps)",
        ""
    ])

    # Intellectual gaps
    intellectual_gaps = gap_analysis.get("intellectual_gaps", [])
    if intellectual_gaps:
        md_lines.append("### Intellectual Gaps (Untested Assumptions & Contradictions)")
        for i, gap in enumerate(intellectual_gaps, 1):
            md_lines.append(f"{i}. {gap}")
        md_lines.append("")

    # Methodological gaps
    methodological_gaps = gap_analysis.get("methodological_gaps", [])
    if methodological_gaps:
        md_lines.append("### Methodological Gaps (Measurement & Design Limitations)")
        for i, gap in enumerate(methodological_gaps, 1):
            md_lines.append(f"{i}. {gap}")
        md_lines.append("")

    # Research gaps (deprecated but keep for backward compat)
    research_gaps = gap_analysis.get("research_gaps", [])
    if research_gaps:
        md_lines.append("### Research Gaps (Deprecated)")
        for i, gap in enumerate(research_gaps, 1):
            md_lines.append(f"{i}. {gap}")
        md_lines.append("")

    # Priority recommendations
    priority_recommendations = gap_analysis.get("priority_recommendations", [])
    if priority_recommendations:
        md_lines.append("### Priority Recommendations")
        for i, rec in enumerate(priority_recommendations, 1):
            md_lines.append(f"{i}. {rec}")
        md_lines.append("")

    # Conclusion
    md_lines.extend([
        "---",
        "",
        "## Conclusion",
        "",
        f"This analysis identified multiple promising research directions based on {len(cluster_analyses)} clusters of literature. ",
        f"The highest priority recommendations focus on {priority_recommendations[0] if priority_recommendations else 'unexplored areas'}.",
        "",
        f"*Generated by Knowledge Graph Builder on {datetime.now().strftime('%Y-%m-%d')}*"
    ])

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print(f"✓ Markdown report saved: {filepath}")
    return str(filepath)
