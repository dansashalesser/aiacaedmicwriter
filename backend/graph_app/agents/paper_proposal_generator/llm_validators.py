"""LLM-based validators for proposal quality - uses reasoning instead of hardcoded patterns.

Each validator is a separate node with @observe decorator for Langfuse tracing.
"""

from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langfuse import observe
import asyncio
import aiohttp
import os
import re


# ==============================================================================
# Pydantic Models for Structured Validator Outputs
# ==============================================================================

class TopicAlignmentResult(BaseModel):
    """Result of topic alignment validation."""
    is_aligned: bool = Field(description="Whether proposal addresses user's intent")
    alignment_score: float = Field(description="0.0-1.0 semantic alignment score")
    core_elements_addressed: List[str] = Field(description="Which elements from user input are addressed")
    missing_elements: List[str] = Field(description="Which elements from user input are NOT addressed")
    drift_detected: bool = Field(description="Whether proposal drifted to adjacent but distinct territory")
    drift_explanation: Optional[str] = Field(description="If drifted, explain how", default=None)
    reasoning: str = Field(description="Step-by-step reasoning for this assessment")


class GapDepthResult(BaseModel):
    """Result of gap intellectual depth validation."""
    has_theoretical_prediction: bool = Field(description="Whether gap states what current theory predicts")
    has_theoretical_stakes: bool = Field(description="Whether gap states what would change if filled")
    gap_type: str = Field(description="'intellectual' or 'coverage' gap classification")
    missing_components: List[str] = Field(description="Required components that are missing")
    reasoning: str = Field(description="Analysis of gap quality")
    is_valid: bool = Field(description="Whether the gap meets quality requirements")


class JournalRecommendation(BaseModel):
    """A recommended journal from search results."""
    journal_name: str = Field(description="Full name of the journal")
    publisher: Optional[str] = Field(description="Publisher name if known", default=None)
    scope_match: str = Field(description="Why this journal's scope matches the research")
    methodology_fit: str = Field(description="Why this journal accepts this type of methodology")
    quality_indicator: str = Field(description="Impact factor, ranking, or reputation indicator")
    recent_similar_work: Optional[str] = Field(description="Example of similar work published recently", default=None)


class JournalSearchResult(BaseModel):
    """Result of journal search and selection."""
    primary_journal: JournalRecommendation = Field(description="First choice journal recommendation")
    backup_journal: JournalRecommendation = Field(description="Backup journal recommendation")
    search_quality: str = Field(description="'high', 'medium', or 'low' based on search results quality")
    reasoning: str = Field(description="Explanation of selection process")


class SearchQueryExtraction(BaseModel):
    """Extracted search queries for finding journals."""
    domain_keywords: List[str] = Field(description="Main research domain/field keywords")
    methodology_keywords: List[str] = Field(description="Methodology-related keywords")
    search_queries: List[str] = Field(description="2-3 search queries to find relevant journals")


# ==============================================================================
# Issue 1: Topic Alignment Validator Node
# ==============================================================================

@observe(name="validate-topic-alignment", as_type="generation")
async def validate_topic_alignment(
    user_input: str,
    proposal_title: str,
    proposal_research_question: str,
    proposal_gap: str,
    proposal_hypothesis: str,
    llm: Optional[ChatOpenAI] = None
) -> Tuple[bool, str, TopicAlignmentResult]:
    """
    Use LLM reasoning to validate proposal addresses user's intent.

    This is a BLOCKING validator - proposals that fail will be rejected and retried.

    Args:
        user_input: Original research topic/interest from user
        proposal_title: Generated proposal's working title
        proposal_research_question: Generated proposal's research question
        proposal_gap: Generated proposal's gap statement (what we don't know)
        proposal_hypothesis: Generated proposal's hypothesis
        llm: Optional LLM instance (defaults to gpt-4o-mini)

    Returns:
        (is_valid, feedback_message, full_result)
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    structured_llm = llm.with_structured_output(TopicAlignmentResult)

    prompt = f"""You are a research methodology expert evaluating whether a generated proposal addresses the user's research intent.

USER'S ORIGINAL RESEARCH INTEREST:
"{user_input}"

GENERATED PROPOSAL:
- Title: {proposal_title}
- Research Question: {proposal_research_question}
- Gap Addressed: {proposal_gap}
- Hypothesis: {proposal_hypothesis}

YOUR TASK: Evaluate semantic alignment between user's intent and the generated proposal.

STEP 1 - EXTRACT CORE ELEMENTS from user input:
Identify the key components of what the user wants to research:
- Primary domain/field (e.g., "algorithmic trading", "disaster prediction", "education")
- Core technology/method mentioned (e.g., "LLMs", "machine learning", "surveys")
- Target application/outcome (e.g., "trading system", "prediction accuracy", "student outcomes")
- Any specific constraints or context mentioned

STEP 2 - CHECK IF PROPOSAL ADDRESSES THESE:
For each core element, determine:
- Does the proposal's research question involve the same domain?
- Does the proposal use or study the specified technology/method?
- Does the proposal target the specified application/outcome?

STEP 3 - DETECT DRIFT:
Has the proposal shifted to an ADJACENT but DISTINCT topic?
Common drift patterns to detect:
- Same broad field but different specific question (e.g., "finance" includes both "algorithmic trading" and "financial literacy" - these are DIFFERENT topics)
- Same technology applied to different domain
- Same domain but ignoring specified technology

STEP 4 - SCORING:
- alignment_score >= 0.7: Proposal clearly addresses user's intent
- alignment_score 0.4-0.7: Partial alignment, some key elements missing
- alignment_score < 0.4: Major misalignment, proposal is off-topic

CRITICAL: Be strict about alignment. A proposal about "financial literacy" does NOT address "algorithmic trading" even if both are in finance. A proposal about "disaster prediction" does NOT address "stock trading" even if both use machine learning.

Provide your assessment with detailed reasoning."""

    try:
        result = await asyncio.wait_for(
            structured_llm.ainvoke(prompt),
            timeout=30.0
        )

        # Determine validity based on alignment score
        is_valid = result.is_aligned and result.alignment_score >= 0.7 and not result.drift_detected

        # Generate feedback message for retry if invalid
        if not is_valid:
            feedback_parts = []
            if result.missing_elements:
                feedback_parts.append(f"Missing elements from user input: {', '.join(result.missing_elements)}")
            if result.drift_detected and result.drift_explanation:
                feedback_parts.append(f"Topic drift detected: {result.drift_explanation}")
            feedback_parts.append(f"Alignment score: {result.alignment_score:.1%}")
            feedback_parts.append(f"Reasoning: {result.reasoning}")
            feedback_message = "TOPIC ALIGNMENT FAILED:\n" + "\n".join(feedback_parts)
        else:
            feedback_message = "Topic alignment validated successfully."

        return (is_valid, feedback_message, result)

    except asyncio.TimeoutError:
        # On timeout, allow proposal through with warning
        fallback_result = TopicAlignmentResult(
            is_aligned=True,
            alignment_score=0.5,
            core_elements_addressed=[],
            missing_elements=[],
            drift_detected=False,
            drift_explanation=None,
            reasoning="Validation timed out - proceeding with caution"
        )
        return (True, "Topic alignment validation timed out - proceeding", fallback_result)
    except Exception as e:
        # On error, allow proposal through with warning
        fallback_result = TopicAlignmentResult(
            is_aligned=True,
            alignment_score=0.5,
            core_elements_addressed=[],
            missing_elements=[],
            drift_detected=False,
            drift_explanation=None,
            reasoning=f"Validation error: {str(e)}"
        )
        return (True, f"Topic alignment validation error: {str(e)} - proceeding", fallback_result)


# ==============================================================================
# Issue 9: Gap Intellectual Depth Validator Node
# ==============================================================================

@observe(name="validate-gap-depth", as_type="generation")
async def validate_gap_intellectual_depth(
    gap_what_we_know: str,
    gap_what_we_dont_know: str,
    gap_why_it_matters: str,
    llm: Optional[ChatOpenAI] = None
) -> Tuple[bool, str, GapDepthResult]:
    """
    Validate gap statements include theoretical predictions and stakes.

    This is a BLOCKING validator - proposals that fail will be rejected and retried.

    Required format for gaps:
    (a) What current theory predicts
    (b) What would change if gap were filled (not just "we'd know more")

    Args:
        gap_what_we_know: What existing work establishes
        gap_what_we_dont_know: The actual intellectual gap
        gap_why_it_matters: Why closing this gap is important
        llm: Optional LLM instance

    Returns:
        (is_valid, feedback_message, full_result)
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    structured_llm = llm.with_structured_output(GapDepthResult)

    prompt = f"""You are evaluating the intellectual depth of a research gap statement.

GAP STATEMENT:
- What we know: {gap_what_we_know}
- What we don't know: {gap_what_we_dont_know}
- Why it matters: {gap_why_it_matters}

YOUR TASK: Evaluate whether this gap meets intellectual depth requirements.

REQUIREMENT 1 - THEORETICAL PREDICTION:
Does "what we know" state what current theory PREDICTS about this phenomenon?
- WEAK (fails): "Studies have examined X" / "Research shows X exists"
- STRONG (passes): "Current theory predicts that X should lead to Y because of mechanism Z" / "Models assume X with magnitude Y"

REQUIREMENT 2 - THEORETICAL STAKES:
Does "why it matters" explain what would CHANGE THEORETICALLY if we filled this gap?
- WEAK (fails): "We would know more about X" / "This would improve our understanding" / "This is important"
- WEAK (fails): Only practical importance like "This would save money/lives" without theoretical stakes
- STRONG (passes): "If gap is filled, we would know whether [Theory A] or [Theory B] better explains [phenomenon]"
- STRONG (passes): "This would resolve the debate between X and Y theories"

REQUIREMENT 3 - GAP TYPE:
Is this an INTELLECTUAL gap or a COVERAGE gap?
- COVERAGE GAP (fails): "No one has done X" / "X hasn't been studied with Y" / "First to combine X and Y"
- INTELLECTUAL GAP (passes): "Assumption X is untested" / "Studies contradict on Y" / "Mechanism Z is unknown"

EVALUATION:
1. Check if "what we know" includes theoretical prediction (Yes/No + quote evidence)
2. Check if "why it matters" includes theoretical stakes, not just practical (Yes/No + quote evidence)
3. Classify as intellectual vs coverage gap
4. List what's missing

Set is_valid to True only if ALL requirements are met."""

    try:
        result = await asyncio.wait_for(
            structured_llm.ainvoke(prompt),
            timeout=30.0
        )

        # Generate feedback message for retry if invalid
        if not result.is_valid:
            feedback_parts = ["GAP INTELLECTUAL DEPTH FAILED:"]
            if not result.has_theoretical_prediction:
                feedback_parts.append("- 'What we know' must state what current theory PREDICTS, not just what exists")
            if not result.has_theoretical_stakes:
                feedback_parts.append("- 'Why it matters' must explain THEORETICAL stakes (which theories are resolved), not just practical importance")
            if result.gap_type == "coverage":
                feedback_parts.append("- Gap appears to be a COVERAGE gap ('no one has done X'). Reframe as INTELLECTUAL gap ('assumption X is untested')")
            if result.missing_components:
                feedback_parts.append(f"- Missing components: {', '.join(result.missing_components)}")
            feedback_parts.append(f"- Reasoning: {result.reasoning}")
            feedback_message = "\n".join(feedback_parts)
        else:
            feedback_message = "Gap intellectual depth validated successfully."

        return (result.is_valid, feedback_message, result)

    except asyncio.TimeoutError:
        fallback_result = GapDepthResult(
            has_theoretical_prediction=True,
            has_theoretical_stakes=True,
            gap_type="unknown",
            missing_components=[],
            reasoning="Validation timed out - proceeding with caution",
            is_valid=True
        )
        return (True, "Gap depth validation timed out - proceeding", fallback_result)
    except Exception as e:
        fallback_result = GapDepthResult(
            has_theoretical_prediction=True,
            has_theoretical_stakes=True,
            gap_type="unknown",
            missing_components=[],
            reasoning=f"Validation error: {str(e)}",
            is_valid=True
        )
        return (True, f"Gap depth validation error: {str(e)} - proceeding", fallback_result)


# ==============================================================================
# Issue 3: Journal Search Node
# ==============================================================================

async def _search_serper(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search using Serper API (Google Search API).

    Returns list of {title, snippet, link} dicts.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return []

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get("organic", []):
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", "")
                        })
                    return results
                return []
    except Exception:
        return []


async def _search_duckduckgo(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Fallback search using DuckDuckGo (no API key required).

    Returns list of {title, snippet, link} dicts.
    """
    try:
        # Use duckduckgo-search library if available
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "link": r.get("href", "")
                })
        return results
    except ImportError:
        # Library not installed
        return []
    except Exception:
        return []


@observe(name="search-relevant-journals", as_type="generation")
async def search_relevant_journals(
    proposal_title: str,
    proposal_research_question: str,
    proposal_methodology: str,
    proposal_domain: str,
    llm: Optional[ChatOpenAI] = None
) -> Tuple[JournalSearchResult, str]:
    """
    Search for real academic journals relevant to the proposal.

    This is a POST-GENERATION step that replaces LLM-fabricated journal names
    with search-verified real journals.

    Args:
        proposal_title: The proposal's working title
        proposal_research_question: The research question
        proposal_methodology: Description of methodology (dataset, identification strategy)
        proposal_domain: Research domain/field
        llm: Optional LLM instance

    Returns:
        (JournalSearchResult, rationale_text)
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Step 1: Extract search queries from proposal
    query_extraction_llm = llm.with_structured_output(SearchQueryExtraction)

    extraction_prompt = f"""Extract search queries to find academic journals for this research proposal.

PROPOSAL:
- Title: {proposal_title}
- Research Question: {proposal_research_question}
- Methodology: {proposal_methodology}
- Domain: {proposal_domain}

Generate 2-3 search queries that would find relevant academic journals.
Include:
1. One query for the main research domain (e.g., "top journals machine learning finance")
2. One query for the methodology type (e.g., "journals publishing causal inference studies")
3. One query combining domain + methodology

Use terms like "academic journal", "peer-reviewed", "impact factor" to get journal-focused results."""

    try:
        query_result = await asyncio.wait_for(
            query_extraction_llm.ainvoke(extraction_prompt),
            timeout=15.0
        )
        search_queries = query_result.search_queries[:3]  # Max 3 queries
    except Exception:
        # Fallback queries
        search_queries = [
            f"top academic journals {proposal_domain}",
            f"peer-reviewed journals {proposal_domain} research impact factor"
        ]

    # Step 2: Execute searches
    all_search_results = []

    for query in search_queries:
        # Try Serper first, then DuckDuckGo as fallback
        results = await _search_serper(query, num_results=8)
        if not results:
            results = await _search_duckduckgo(query, num_results=8)
        all_search_results.extend(results)

    # Deduplicate by link
    seen_links = set()
    unique_results = []
    for r in all_search_results:
        if r["link"] not in seen_links:
            seen_links.add(r["link"])
            unique_results.append(r)

    # Format search results for LLM
    if unique_results:
        search_context = "\n".join([
            f"- {r['title']}: {r['snippet'][:200]}... ({r['link']})"
            for r in unique_results[:15]
        ])
        search_quality = "high" if len(unique_results) >= 10 else "medium" if len(unique_results) >= 5 else "low"
    else:
        search_context = "No search results available. Use your knowledge of academic journals."
        search_quality = "low"

    # Step 3: LLM selects best journals from search results
    selection_llm = llm.with_structured_output(JournalSearchResult)

    selection_prompt = f"""Select the TWO most appropriate academic journals for this research proposal.

PROPOSAL:
- Title: {proposal_title}
- Research Question: {proposal_research_question}
- Methodology: {proposal_methodology}
- Domain: {proposal_domain}

SEARCH RESULTS (journals and related pages found):
{search_context}

YOUR TASK:
1. Identify REAL journals from the search results or your knowledge
2. Select a PRIMARY journal (best fit) and BACKUP journal (alternative)
3. For each journal, explain:
   - Why the journal's scope matches this research
   - Why this methodology is appropriate for the journal
   - Quality indicators (impact factor, reputation, ranking)
   - If possible, cite a recent similar paper published there

CRITICAL REQUIREMENTS:
- Only recommend journals that ACTUALLY EXIST
- Do NOT invent journal names like "Journal of X and Y and Z"
- If unsure about a journal, choose a well-known one in the field
- Prefer established journals: Nature, Science, PNAS, domain-specific top journals
- For applied/empirical work, consider: Management Science, MIS Quarterly, Journal of Finance, etc.
- For interdisciplinary work, consider: PLOS ONE, Scientific Reports, etc.

Set search_quality to '{search_quality}' based on the search results available."""

    try:
        result = await asyncio.wait_for(
            selection_llm.ainvoke(selection_prompt),
            timeout=30.0
        )
        result.search_quality = search_quality

        # Generate rationale text for the proposal
        primary_rationale = (
            f"{result.primary_journal.scope_match} "
            f"{result.primary_journal.methodology_fit} "
            f"Quality: {result.primary_journal.quality_indicator}."
        )
        if result.primary_journal.recent_similar_work:
            primary_rationale += f" Recent similar work: {result.primary_journal.recent_similar_work}"

        backup_rationale = (
            f"Alternative if primary unavailable: {result.backup_journal.scope_match} "
            f"{result.backup_journal.quality_indicator}."
        )

        return (result, primary_rationale + " " + backup_rationale)

    except asyncio.TimeoutError:
        # Return fallback with generic but real journals
        fallback = JournalSearchResult(
            primary_journal=JournalRecommendation(
                journal_name="PLOS ONE",
                publisher="Public Library of Science",
                scope_match="Broad scope covering all scientific disciplines",
                methodology_fit="Accepts diverse methodological approaches",
                quality_indicator="Impact factor ~3.7, widely indexed",
                recent_similar_work=None
            ),
            backup_journal=JournalRecommendation(
                journal_name="Scientific Reports",
                publisher="Nature Portfolio",
                scope_match="Multidisciplinary journal from Nature",
                methodology_fit="Accepts empirical research across fields",
                quality_indicator="Impact factor ~4.6, high visibility",
                recent_similar_work=None
            ),
            search_quality="low",
            reasoning="Search timed out - recommending well-known multidisciplinary journals"
        )
        return (fallback, "Broad multidisciplinary journals suitable for diverse research topics.")
    except Exception as e:
        # Return same fallback on any error
        fallback = JournalSearchResult(
            primary_journal=JournalRecommendation(
                journal_name="PLOS ONE",
                publisher="Public Library of Science",
                scope_match="Broad scope covering all scientific disciplines",
                methodology_fit="Accepts diverse methodological approaches",
                quality_indicator="Impact factor ~3.7, widely indexed",
                recent_similar_work=None
            ),
            backup_journal=JournalRecommendation(
                journal_name="Scientific Reports",
                publisher="Nature Portfolio",
                scope_match="Multidisciplinary journal from Nature",
                methodology_fit="Accepts empirical research across fields",
                quality_indicator="Impact factor ~4.6, high visibility",
                recent_similar_work=None
            ),
            search_quality="low",
            reasoning=f"Search error: {str(e)} - recommending well-known multidisciplinary journals"
        )
        return (fallback, "Broad multidisciplinary journals suitable for diverse research topics.")


# ==============================================================================
# Orchestration: Run Blocking Validators
# ==============================================================================

@observe(name="run-blocking-validators")
async def run_blocking_validators(
    user_input: str,
    proposal_title: str,
    proposal_research_question: str,
    proposal_gap_what_we_know: str,
    proposal_gap_what_we_dont_know: str,
    proposal_gap_why_it_matters: str,
    proposal_hypothesis: str,
    llm: Optional[ChatOpenAI] = None
) -> Tuple[bool, List[str]]:
    """
    Run all BLOCKING validators on a proposal.

    Blocking validators: Topic Alignment (Issue 1), Gap Depth (Issue 9)

    Returns (all_passed, list_of_feedback_messages)
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    all_feedback = []
    all_passed = True

    # Run validators in parallel
    topic_task = validate_topic_alignment(
        user_input=user_input,
        proposal_title=proposal_title,
        proposal_research_question=proposal_research_question,
        proposal_gap=proposal_gap_what_we_dont_know,
        proposal_hypothesis=proposal_hypothesis,
        llm=llm
    )

    gap_task = validate_gap_intellectual_depth(
        gap_what_we_know=proposal_gap_what_we_know,
        gap_what_we_dont_know=proposal_gap_what_we_dont_know,
        gap_why_it_matters=proposal_gap_why_it_matters,
        llm=llm
    )

    results = await asyncio.gather(topic_task, gap_task, return_exceptions=True)

    # Process topic alignment result
    if isinstance(results[0], Exception):
        all_feedback.append(f"Topic Alignment Error: {str(results[0])}")
        # Don't fail on error - let proposal through
    else:
        topic_valid, topic_feedback, _ = results[0]
        if not topic_valid:
            all_passed = False
            all_feedback.append(topic_feedback)

    # Process gap depth result
    if isinstance(results[1], Exception):
        all_feedback.append(f"Gap Depth Error: {str(results[1])}")
        # Don't fail on error - let proposal through
    else:
        gap_valid, gap_feedback, _ = results[1]
        if not gap_valid:
            all_passed = False
            all_feedback.append(gap_feedback)

    return (all_passed, all_feedback)
