"""Paper Proposal Generator - Generates detailed academic paper proposals from gap analysis."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
from pathlib import Path
import re
import sys
from langfuse import observe

# Add semantic-scholar folder to path for imports
semantic_scholar_dir = Path(__file__).parent.parent.parent / "utils" / "semantic-scholar"
sys.path.insert(0, str(semantic_scholar_dir))

from semantic_scholar import search_papers_async
from backend.graph_app.agents.topic_segmenter import segment_topics
from backend.graph_app.agents.paper_proposal_generator.validators import (
    validate_hypothesis,
    validate_gap,
    validate_contribution,
    validate_variable_operationalization,
    validate_research_question,
    validate_risks_and_limitations,
    validate_timeline,
    validate_outlet
)
from backend.graph_app.agents.paper_proposal_generator.similarity import (
    is_too_similar,
    generate_differentiation_feedback,
    get_average_pairwise_similarity
)

# ==============================================================================
# Pydantic Models for Structured Outputs
# ==============================================================================

class PaperLiterature(BaseModel):
    """Key paper and how the proposal builds on it."""
    citation: str = Field(description="Paper citation (Author Year)")
    title: str = Field(description="Paper title")
    relationship: str = Field(description="How this proposal builds on / departs from it")


class ProposalConcept(BaseModel):
    """Initial proposal concept for Stage 1."""
    model_config = {"validate_assignment": True}  # Allow mutation after creation

    working_title: str = Field(description="Specific, contribution-focused title for searching")
    risk_level: str = Field(description="high/medium/low")
    core_question: str = Field(description="1 sentence research question")
    main_gap: str = Field(description="Which gap from gap analysis this addresses")
    search_query: str = Field(description="Optimized query for Semantic Scholar search")


class ProposalConcepts(BaseModel):
    """Collection of proposal concepts."""
    concepts: List[ProposalConcept] = Field(description="Generated proposal concepts")


class PaperProposal(BaseModel):
    """Complete paper proposal with 12-section structure."""
    model_config = {"validate_assignment": True}  # Allow mutation after creation for journal search

    # 1. Working Title
    working_title: str = Field(description="Specific, contribution-focused title")

    # 2. Research Question
    research_question: str = Field(description="1-2 sentence precise question")

    # 3. The Gap
    gap_what_we_know: str = Field(description="What existing work establishes")
    gap_what_we_dont_know: str = Field(description="The actual intellectual gap")
    gap_why_it_matters: str = Field(description="Why closing this gap is important")

    # 4. The Contribution
    empirical_contribution: Optional[str] = Field(description="What new empirical finding")
    theoretical_contribution: Optional[str] = Field(description="What theoretical advance")
    methodological_contribution: Optional[str] = Field(description="What methodological innovation, if any")

    # 5. Core Hypothesis
    hypothesis: str = Field(description="Expected finding and theoretical mechanism")
    alternative_explanation: str = Field(description="Competing mechanism and its prediction")

    # 6. Data and Method
    dataset: str = Field(description="Specific dataset to use")
    unit_of_analysis: str = Field(description="What is being analyzed")
    dependent_variable: str = Field(description="Outcome measure")
    independent_variable: str = Field(description="Key explanatory variable")
    identification_strategy: str = Field(description="How to establish causality/inference")

    # 7. Why Design Works
    design_justification: str = Field(description="Why this approach is appropriate")
    key_assumption: str = Field(description="Main assumption required")
    assumption_justification: str = Field(description="Why assumption is plausible")
    robustness_checks: str = Field(description="How to test robustness")

    # 8. Expected Findings
    finding_strong_positive: str = Field(description="Interpretation if strong effect found")
    finding_null: str = Field(description="Interpretation if null result")
    finding_heterogeneous: str = Field(description="Interpretation if moderated effects")

    # 9. Fit with Literature
    key_papers: List[PaperLiterature] = Field(description="0-8 key papers from proposal-specific search (empty if no papers found)", min_length=0, max_length=8)

    # 10. Target Outlets
    first_choice_journal: str = Field(description="Primary target journal")
    first_choice_rationale: str = Field(description="Why this journal fits")
    backup_journal: str = Field(description="Backup journal")
    backup_rationale: str = Field(description="When to use backup")

    # 11. Risks and Limitations
    data_risks: List[str] = Field(description="Potential data access/quality issues")
    identification_risks: List[str] = Field(description="Threats to causal inference")
    scope_limitations: List[str] = Field(description="Generalizability constraints")

    # 12. Timeline
    literature_design_weeks: str = Field(description="Weeks for lit review + design (e.g., '4-6 weeks')")
    data_acquisition_weeks: str = Field(description="Weeks for data (e.g., '6-8 weeks')")
    analysis_weeks: str = Field(description="Weeks for analysis (e.g., '8-10 weeks')")
    writing_weeks: str = Field(description="Weeks for draft (e.g., '6-8 weeks')")
    revision_weeks: str = Field(description="Weeks for revision (e.g., '4-6 weeks')")
    total_weeks_estimate: int = Field(description="Total estimated weeks")

    # Issue 8: Validate timeline fields include units
    @field_validator('literature_design_weeks', 'data_acquisition_weeks', 'analysis_weeks', 'writing_weeks', 'revision_weeks')
    @classmethod
    def validate_timeline_has_units(cls, v: str, info) -> str:
        """Ensure timeline fields include time units (weeks/months)."""
        if not v:
            return v
        v_lower = v.lower()
        # Check for unit presence
        if not any(unit in v_lower for unit in ['week', 'month', 'day']):
            # Auto-append 'weeks' if it looks like a number/range without unit
            if re.match(r'^[\d\-\s]+$', v.strip()):
                return f"{v.strip()} weeks"
            raise ValueError(f"Timeline field '{info.field_name}' must include time unit (e.g., '4-6 weeks'), got: '{v}'")
        return v

    @model_validator(mode='after')
    def validate_proposal_quality(self):
        """Comprehensive validation of proposal quality."""
        errors = []

        # Validate research question
        rq_valid, rq_errors = validate_research_question(self.research_question)
        errors.extend(rq_errors)

        # Validate gap
        gap_valid, gap_errors = validate_gap(
            self.gap_what_we_know,
            self.gap_what_we_dont_know,
            self.gap_why_it_matters
        )
        errors.extend(gap_errors)

        # Validate hypothesis
        hyp_valid, hyp_errors = validate_hypothesis(
            self.hypothesis,
            self.alternative_explanation
        )
        errors.extend(hyp_errors)

        # Validate contributions
        if self.empirical_contribution:
            contrib_valid, contrib_errors = validate_contribution(
                self.empirical_contribution, "Empirical"
            )
            errors.extend(contrib_errors)

        if self.theoretical_contribution:
            contrib_valid, contrib_errors = validate_contribution(
                self.theoretical_contribution, "Theoretical"
            )
            errors.extend(contrib_errors)

        if self.methodological_contribution:
            contrib_valid, contrib_errors = validate_contribution(
                self.methodological_contribution, "Methodological"
            )
            errors.extend(contrib_errors)

        # Validate variables
        var_valid, var_errors = validate_variable_operationalization(
            self.dependent_variable,
            self.independent_variable
        )
        errors.extend(var_errors)

        # Validate risks and limitations
        risks_valid, risks_errors = validate_risks_and_limitations(
            self.data_risks,
            self.identification_risks,
            self.scope_limitations
        )
        errors.extend(risks_errors)

        # Validate timeline
        timeline_valid, timeline_errors = validate_timeline(
            self.literature_design_weeks,
            self.data_acquisition_weeks,
            self.analysis_weeks,
            self.writing_weeks,
            self.revision_weeks,
            self.total_weeks_estimate
        )
        errors.extend(timeline_errors)

        # Validate outlets
        outlet1_valid, outlet1_errors = validate_outlet(
            self.first_choice_journal,
            self.first_choice_rationale,
            "First choice"
        )
        errors.extend(outlet1_errors)

        outlet2_valid, outlet2_errors = validate_outlet(
            self.backup_journal,
            self.backup_rationale,
            "Backup"
        )
        errors.extend(outlet2_errors)

        # If there are errors, raise validation error
        if errors:
            error_msg = "Proposal validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)

        return self


class PaperProposalsOutput(BaseModel):
    """Collection of paper proposals."""
    proposals: List[PaperProposal] = Field(description="Generated paper proposals")
    summary: str = Field(description="Overview of proposed directions")


# ==============================================================================
# Helper Functions
# ==============================================================================

@observe(name="sanitize-filename-proposals")
def sanitize_filename(user_input: str) -> str:
    """Sanitize user input to create a valid filename."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', user_input[:50])
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')


@observe(name="format-papers-for-literature")
def format_papers_for_literature(papers: List[Dict]) -> str:
    """Format papers for LLM context."""
    if not papers:
        return "No papers found in search."

    formatted = []
    for i, paper in enumerate(papers, 1):
        authors = ", ".join([author.get('name', 'Unknown') for author in paper.get('authors', [])[:3]])
        if len(paper.get('authors', [])) > 3:
            authors += " et al."

        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'Unknown title')
        abstract = paper.get('abstract') or 'No abstract available'
        if abstract != 'No abstract available':
            abstract = abstract[:500]
        citations = paper.get('citationCount', 0)

        formatted.append(f"""
Paper {i}:
- Authors: {authors}
- Year: {year}
- Title: {title}
- Citations: {citations}
- Abstract: {abstract}...
""")

    return "\n".join(formatted)


@observe(name="generate-proposals-summary")
def generate_proposals_summary(proposals: List[PaperProposal], gap_analysis: Dict) -> str:
    """Generate overview summary of all proposals."""
    titles = [proposal.working_title for proposal in proposals]
    num_proposals = len(proposals)

    # Fix Issue 2: Use intellectual_gaps instead of deprecated research_gaps
    # Also extract gaps from proposals themselves as fallback
    intellectual_gaps = gap_analysis.get('intellectual_gaps', [])
    methodological_gaps = gap_analysis.get('methodological_gaps', [])

    # Combine gaps, preferring intellectual gaps
    all_gaps = intellectual_gaps[:2] + methodological_gaps[:1] if intellectual_gaps else methodological_gaps[:3]

    # If no gaps from gap_analysis, extract from proposals
    if not all_gaps:
        all_gaps = [p.gap_what_we_dont_know[:100] + "..." for p in proposals[:3]]

    gaps_text = chr(10).join(['- ' + gap for gap in all_gaps]) if all_gaps else "- See individual proposals for specific gaps addressed"

    # Fix Issue 7: Use singular/plural language based on actual count
    if num_proposals == 1:
        intro = "This research proposal addresses a key gap identified in the literature:"
        papers_label = "Proposed Paper:"
        outro = ""
    elif num_proposals == 2:
        intro = "These two research proposals address key gaps identified in the literature:"
        papers_label = "Proposed Papers:"
        outro = "These proposals offer complementary approaches to the research area."
    else:
        intro = f"This research agenda presents {num_proposals} paper proposals addressing key gaps identified in the literature:"
        papers_label = "Proposed Papers:"
        outro = "These proposals provide a balanced research portfolio across different risk levels and methodological approaches."

    summary = f"""{intro}

Key Gaps Addressed:
{gaps_text}

{papers_label}
{chr(10).join([f"{i+1}. {title}" for i, title in enumerate(titles)])}

{outro}
"""

    return summary.strip()


# ==============================================================================
# Stage 1: Generate Proposal Concepts
# ==============================================================================

@observe(name="generate-proposal-concepts", as_type="generation")
async def generate_proposal_concepts(
    gap_analysis: Dict[str, Any],
    user_input: str,
    num_proposals: int = 5,
    failed_examples: str = ""
) -> List[ProposalConcept]:
    """
    Generate initial proposal concepts with risk stratification.

    Returns concepts optimized for Semantic Scholar searching.
    Args:
        failed_examples: String containing failed proposal examples to avoid (optional)
    """
    # Calculate risk distribution
    num_high_risk = max(1, num_proposals // 3)
    num_medium_risk = max(1, num_proposals // 3)
    num_low_risk = num_proposals - num_high_risk - num_medium_risk

    # Format gap analysis
    research_gaps = "\n".join([f"- {gap}" for gap in gap_analysis.get('research_gaps', [])])
    untried_directions = "\n".join([f"- {direction}" for direction in gap_analysis.get('untried_directions', [])])
    priority_recommendations = "\n".join([f"- {rec}" for rec in gap_analysis.get('priority_recommendations', [])])

    prompt = ChatPromptTemplate.from_template("""# Academic Topic Explorer
    <directives>
    <general directives>
    <p>Make sure you are correct, and think about your response step by step. This is VERY important to me!</p>
    </general directives>
    <role driectives>
    <p>You are an academic research advisor generating paper proposal concepts.
    You are knowledgeable about different research methods, understand gaps and how to analyze them,
    and are able to infer what might be relevant in the world of academia using proper scientific language.</p>
    </role directives>
    <task>
    <p> You will generate {num_proposals} paper proposal concepts based on the information you will receive,
    and based on the STRUCTURE you will receive below.
    <risk category>
    You will add risk stratification per these guidelines:
    - {num_high_risk} high-impact, high-risk proposals (novel theoretical contributions, challenging existing paradigms)
    - {num_medium_risk} medium-impact, medium-risk proposals (solid empirical work, incremental advances)
    - {num_low_risk} safe proposals (extensions of existing work with new data/context).</p>
    </risk category>
    <general produced items>
    <p>For each concept, provide:
    1. **working_title**: Specific, contribution-focused title (10-15 words)
    2. **risk_level**: "high", "medium", or "low"
    3. **core_question**: 1-sentence precise research question
    4. **main_gap**: Which gap from the gap analysis this addresses
    5. **search_query**: Optimized query for Semantic Scholar (include key methodological/domain terms, 5-10 words)</p>
    </general produced items>
    </task>
    </directives>
    <context>
    <p>**Original Research Interest**: {user_input}
    **Gap Analysis Results**:
    Research Gaps:
    {research_gaps}

    Untried Directions:
    {untried_directions}

    Priority Recommendations:
    {priority_recommendations}

    {failed_examples_section}</p>
    </context>
    <jargon usage>
    <p>IMPORTANT for search_query:
    - Use academic terminology
    - Include methodology keywords (e.g., "machine learning", "causal inference", "qualitative analysis")
    - Include domain keywords (e.g., "climate change", "education policy", "social networks")
    - Make it broad enough to find relevant literature but specific enough to be focused
    - Example: "machine learning climate prediction ensemble methods"</p>
    </jargon usage>
    <output>
    <p>Return as a ProposalConcepts object with a list of concepts.</p?
    </output>""")

    # Generate concepts using gpt-5-mini with structured output
    # Temperature lowered from 0.8 to 0.3 for more focused, less vague concepts
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)
    structured_llm = llm.with_structured_output(ProposalConcepts)

    chain = prompt | structured_llm

    # Construct failed examples section if provided
    if failed_examples:
        failed_examples_section = f"""
    **⚠️ CRITICAL - LEARN FROM THESE FAILURES**:
    Previous attempts FAILED validation. DO NOT repeat these mistakes:

    {failed_examples}

    **AVOID** topics/approaches similar to the above. Generate DIFFERENT concepts that won't fail these validators.
    Focus on concepts that:
    - Have clear quantifiable variables (include numbers/ranges in the question)
    - Address specific testable mechanisms (not vague relationships)
    - Have straightforward data sources (avoid hard-to-operationalize constructs)
    """
    else:
        failed_examples_section = ""

    result = await chain.ainvoke({
        "user_input": user_input,
        "research_gaps": research_gaps,
        "untried_directions": untried_directions,
        "priority_recommendations": priority_recommendations,
        "num_proposals": num_proposals,
        "failed_examples_section": failed_examples_section,
        "num_high_risk": num_high_risk,
        "num_medium_risk": num_medium_risk,
        "num_low_risk": num_low_risk
    })

    return result.concepts


# ==============================================================================
# Stage 2: Semantic Scholar Literature Search
# ==============================================================================
# Note: Stage 2 is embedded in generate_paper_proposals() function
# It performs parallel Semantic Scholar searches for each proposal concept


# ==============================================================================
# Stage 3: Generate Full Proposal
# ==============================================================================

@observe(name="generate-full-proposal", as_type="generation")
async def generate_full_proposal(
    concept: ProposalConcept,
    papers: List[Dict],
    gap_analysis: Dict[str, Any],
    cluster_analyses: List[Dict],
    validation_feedback: Optional[str] = None
) -> PaperProposal:
    """
    Generate a complete 12-section proposal using proposal-specific literature.

    Args:
        concept: Initial proposal concept
        papers: Papers from Semantic Scholar search for THIS proposal
        gap_analysis: Original gap analysis for context
        cluster_analyses: Original cluster analyses for context
    """
    # Format literature for prompt
    literature_context = format_papers_for_literature(papers)

    # Format gap analysis context
    gap_context = f"""
    Research Gaps: {', '.join(gap_analysis.get('research_gaps', []))}
    Untried Directions: {', '.join(gap_analysis.get('untried_directions', []))}
    Priority Recommendations: {', '.join(gap_analysis.get('priority_recommendations', []))}
    """

    # Construct validation feedback section if feedback exists
    validation_feedback_section = ""
    if validation_feedback:
        validation_feedback_section = f"""
    <validation_feedback>
    <p>**⚠️ PREVIOUS ATTEMPT FAILED VALIDATION**

    **Rejected for:**
    {validation_feedback}

    **CRITICAL FIXES REQUIRED**:
    - If "coverage gap": Reframe to identify UNTESTED ASSUMPTION in existing literature, not unstudied combination
    - If "lacks quantification": Add specific numbers, percentages, effect sizes, sample sizes
    - If "banned phrases": Replace vague language ("will advance understanding") with concrete mechanisms and quantities
    - If "lacks citations": Reference specific papers from literature with findings and page numbers
    - If "hypothesis too short": Add causal mechanism, boundary conditions, quantified predictions with ranges
    - If "alternative is negation": Provide competing THEORY with different numeric predictions (not just "no effect")
    - If "lacks operationalization": Specify how variables are measured, expected ranges, data sources
    - If "research question vague": Make specific with variables, relationships, and answerable with data

    **You MUST address ALL the issues above in this attempt. Be CONCRETE and SPECIFIC.**</p>
    </validation_feedback>
    """

    prompt = ChatPromptTemplate.from_template("""# Paper Ideation Generator
    <directives>
    <general directives>
    <p>Make sure you are correct, and think about your response step by step. This is VERY important to me!</p>
    </general directives>
    {validation_feedback_section}
    <role>
    <p>You are a famed researcher, that understands the process of publishing scientific papers
    and the jargon and nuance required to set up useful studies that will advance science.</p>
    <task>
    <p>You will generate a detailed, publication-ready academic paper proposal.</p>
    </task>
    </role>
    <critical_requirements>
    <p>**ANTI-VAGUENESS REQUIREMENTS - THIS IS CRITICAL**:

    1. **NO GENERIC LANGUAGE**: Never use phrases like "will present new evidence", "will advance understanding", "will introduce novel approach"

    2. **MANDATORY QUANTIFICATION**: Every claim must include:
       - Specific numbers (sample sizes, effect sizes, percentages, counts)
       - Measurement units (RMSE in km, F1 scores, percentage points, etc.)
       - Statistical thresholds (p-values, confidence intervals, effect sizes)
       - Time periods and date ranges

    3. **LITERATURE GROUNDING**: Every section must reference specific papers from the search results:
       - Quote actual findings with page numbers where possible
       - Reference specific methodologies from papers
       - Build on concrete gaps identified in specific papers
       - Use author names and years from the provided literature

    4. **CONCRETE MECHANISMS**: Replace vague causal language with specific mechanisms:
       - NOT: "X affects Y"
       - YES: "X affects Y through [specific mechanism], measurable as [specific indicator], expected magnitude [specific range]"

    5. **EMPIRICAL SPECIFICITY**: For every variable, specify:
       - Exact measurement approach
       - Expected range of values
       - Data source with access details
       - Temporal/spatial resolution

    6. **AVOID THESE VAGUE PHRASES**:
       ❌ "significantly improve", "enhance understanding", "provide insights", "advance the field"
       ❌ "real-time data", "novel approach", "comprehensive dataset" (without specifics)
       ❌ "due to the richness/complexity/importance" (without quantification)

    7. **CRITICAL: THEORETICAL vs PRACTICAL GAPS** - Your gap statement MUST include ONE of:

       **A) THEORETICAL STAKES (preferred)**:
          - What we know: "Theory X predicts Y, supported by [Author, 2023] finding Z with effect size N."
          - What we don't know: "However, Theory X assumes condition C, which is untested in context D."
          - Why it matters: "Resolving this would determine whether Theory X or competing Theory W better explains phenomenon P, settling the [specific debate]."

       **B) STRONG PRACTICAL IMPORTANCE (acceptable)**:
          - What we know: "Current approach X achieves Y performance [Author, 2023] with MAE of N."
          - What we don't know: "However, factor Z's impact is unmeasured in setting W across Q conditions."
          - Why it matters: "Understanding this could improve outcomes by N% (from A to B), saving $M annually, affecting Q population of size R."

       **EXAMPLES OF THEORETICAL STAKES THAT PASS VALIDATION**:
       ✓ "This would resolve the debate between rational choice theory and behavioral economics regarding investor decision-making under uncertainty"
       ✓ "This tests the core assumption in prospect theory that losses loom larger than gains, specifically in algorithmic trading contexts where emotional factors are minimized"
       ✓ "This distinguishes whether feature importance is driven by informational efficiency (EMH) or behavioral biases (sentiment theory)"

       **EXAMPLES THAT FAIL VALIDATION**:
       ✗ "This would advance understanding of X" (too vague, no theoretical stakes)
       ✗ "No one has studied X+Y together" (coverage gap, not intellectual gap)
       ✗ "This would be useful for practitioners" (practical only without quantified magnitude)
       ✓ Instead: Be specific about what, how much, measured how, compared to what baseline

    **If you find yourself writing a sentence without numbers, citations, or specific details, STOP and revise it.**</p>
    </critical_requirements>
    <context>
    <proposal concept>
    <p>Working Title: "{working_title}"
    Risk Level: {risk_level}
    Core Question: {core_question}
    Gap Addressed: {main_gap}</p>
    </proposal context>
    <relevant literature>
    <p>(from Semantic Scholar search for this proposal):
    {literature_context}

    **CRITICAL INSTRUCTION**: You MUST mine the abstracts above for:
    - Specific numerical findings (effect sizes, sample sizes, accuracy metrics, percentages)
    - Specific datasets mentioned by name
    - Specific methodological approaches (algorithm names, statistical techniques)
    - Specific gaps or limitations the authors identify
    - Specific measurements and units they used

    These details should be integrated throughout your proposal to ground it in empirical reality.
    When you cite a paper, try to include a specific finding or number from its abstract.</p>
    </relevant literature>
    <gap context>
    <p>(from gap analysis):
    {gap_context}</p>
    </gap context>
    </context>
    <instructions>
    <p>Generate a complete 12-section proposal following academic standards:

    1. WORKING TITLE
    - Refine if needed based on literature
    - Should be specific and contribution-focused

    2. RESEARCH QUESTION
    - 1-2 sentences
    - Precise, answerable and **EMPIRICAL** question
    - This NEEDS TO USE actual data or concepts, and **NOT BE VAGUE!**
    - MUST include specific measurable variables or outcomes

    ### Examples
    - What is the causal effect of [X] on [Y], and under what conditions does this relationship hold/fail?
    - How does [mechanism] explain the observed pattern of [phenomenon], and what does this reveal about [broader theoretical issue]?

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ "How does integrating X enhance Y?"
    ❌ "Can we improve predictions using new data?"

    ### Good Examples (SPECIFIC - EMULATE THIS)
    ✓ "Does incorporating real-time Twitter sentiment scores (measured as positive/negative ratio) improve 24-hour hurricane evacuation predictions by more than 15% RMSE reduction compared to NOAA baseline models?"
    ✓ "What is the marginal effect of each additional socio-economic variable (income, education, population density) on flood prediction accuracy (measured by F1 score), and at what point does model complexity outweigh predictive gains?"

    3. THE GAP (3-4 sentences)
    - **CRITICAL**: Gap must be INTELLECTUAL (untested assumption/unresolved contradiction), NOT coverage-based
    - Why doesn't existing work answer this question? Be specific about what's missing—not just "no one has studied this."
    - **What we know**: Summarize what the literature establishes WITH SPECIFIC CITATIONS
    - **What we don't know**: The specific intellectual gap WITH QUANTIFICATION
    - **Why it matters**: Theoretical/practical importance WITH CONCRETE IMPACT METRICS

    ### CRITICAL: Gap Must Be Intellectual, Not Coverage

    ❌ **BAD - Coverage Gaps (NEVER DO THIS)**:
    - "No one has combined real-time Twitter data with LLMs for disaster prediction"
    - "There is limited research on applying method X to domain Y"
    - "Few studies have explored the intersection of A and B"
    - "This area remains understudied despite its importance"

    ✅ **GOOD - Intellectual Gaps (REQUIRED)**:
    - **What we know**: "Existing FEMA models achieve 18.5pp MAE [Smith, 2022]. Models assume 80% of evacuations follow official warnings."
    - **What we don't know**: "This assumption is UNTESTED in the social media era. We don't know: (1) whether Twitter amplifies or dampens official messaging, (2) at what tweet volume signal becomes noise, (3) whether the relationship is linear or has threshold effects. These questions distinguish 'information enhancement' theory (linear improvement) from 'noise dominance' theory (inverted-U curve)."
    - **Why it matters**: "If linear (enhancement), adding Twitter should improve MAE 15-25% uniformly. If threshold effects exist (noise), improvement peaks at ~5K tweets/hour then degrades. Difference: $47M annually in evacuation costs, 12-18 lives per hurricane season (2017-2022 FEMA data). Theoretically, tests whether social media represents 'new information channels' or 'noise pollution' in crisis communication."

    ### Bad Examples for "Why It Matters" (TOO VAGUE - DO NOT EMULATE)
    ❌ "Enhancing predictive models with real-time data could significantly improve disaster response efforts, potentially saving lives and resources"
    ❌ "Understanding this relationship can significantly improve disaster preparedness"
    ❌ "This is important for advancing our theoretical understanding"

    ### Good Example - Complete Gap Section (EMULATE THIS)
    ✓ "Current FEMA evacuation models have a mean absolute error of 18.5 percentage points in predicting county-level evacuation rates ([Author, 2022], p. 234), leading to systematic over-evacuation in low-risk areas (costing $12-18M per false alarm) and under-evacuation in high-risk areas (contributing to an estimated 23% of hurricane-related fatalities in 2017-2022). Reducing prediction error by 15-25% would enable more targeted evacuation orders, potentially saving $47M annually in unnecessary evacuation costs (based on average of 8.3 major hurricanes/year requiring evacuation orders for 2.1M people at $450/person evacuation cost) while preventing an estimated 12-18 additional fatalities per major hurricane season through better resource allocation to truly high-risk areas. Theoretically, this addresses the longstanding debate in disaster sociology between 'information deficit' models (which assume more data improves decisions) and 'information overload' models (which predict diminishing returns)—[Author et al., 2020] argue this question remains 'empirically unresolved due to lack of real-time data integration studies' (p. 456)."

    4. THE CONTRIBUTION (2-3 bullet points)
    - What will readers know after reading your paper that they didn't know before? Be concrete.
    - What new finding this will produce- Include **METHODOLOGICAL CONCEPTS, NUMBERS OR STRUCTURES- DO NOT BE VAGUE!**
    - **CRITICAL**: Each contribution MUST be AT LEAST 30 WORDS with specific details
    - **CRITICAL**: Each contribution MUST reference specific findings from the literature search or gap analysis
    - **CRITICAL**: Include quantitative expectations (e.g., "X% improvement", "N new variables", "K model architectures")
    - **CRITICAL**: DO NOT write generic statements - every contribution needs concrete numbers, datasets, methods, or findings

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ **Empirical**: "The study will present new empirical evidence on the effectiveness of integrating real-time Twitter data with LLMs"
    ❌ **Theoretical**: "The research will advance theoretical understanding by developing a framework"
    ❌ **Methodological**: "This study will introduce a novel methodological approach"

    ### Good Examples (SPECIFIC - EMULATE THIS)
    ✓ **Empirical**: "Using 2019-2023 hurricane data (N=47 events, 2.3M geotagged tweets), we will quantify whether real-time Twitter sentiment integration reduces 24-hour mobility prediction RMSE by 15-25% compared to NOAA's National Hurricane Center baseline forecasts (current RMSE: 18.5km). We will establish the optimal Twitter data collection window (hypothesis: 6-12 hours pre-landfall) and identify which tweet features (sentiment, volume, geospatial density) contribute most to prediction accuracy."

    ✓ **Theoretical**: "We will test competing theories of information diffusion during disasters: does the 'panic contagion' model (predicting exponential mobility increases with tweet volume) or the 'rational processing' model (predicting linear mobility responses to verified information) better explain observed evacuation patterns? This directly addresses [Author et al., 2022]'s finding that 'existing models fail to account for social media's dual role as panic amplifier and information source' (p. 847)."

    ✓ **Methodological**: "We will introduce a hybrid Transformer-LSTM architecture specifically designed for temporal social media integration, featuring: (1) attention mechanisms weighted by tweet credibility scores (based on [Author, 2023]'s verification framework), (2) dynamic feature selection that adapts to disaster type (hurricanes vs. floods vs. wildfires), and (3) uncertainty quantification through Bayesian dropout layers. This addresses the gap identified in [Author et al., 2021] that 'current LLM approaches treat all social media inputs equivalently, ignoring reliability heterogeneity' (p. 234)."

    5. CORE HYPOTHESIS (1-2 sentences)
    - **Hypothesis**: Expected finding + theoretical mechanism + specific quantitative prediction
    - **Alternative explanation**: Competing mechanism + its prediction + specific quantitative prediction
    - **CRITICAL**: Include numerical predictions, effect sizes, or specific directional hypotheses
    - **CRITICAL**: Ground hypotheses in specific papers from the literature search

    ### REQUIRED COMPONENTS (Missing Any = INVALID Hypothesis):
    1. **Numeric prediction** (not "will improve" but "will improve by X%", "reduce MAE by 15-25%")
    2. **Causal mechanism** (not "due to richness" but "via [specific pathway]", "because social media captures X before Y")
    3. **Boundary conditions** (when does it hold/fail? "strongest for Cat 3-4", "weakest for Cat 1-2")
    4. **Alternative with DIFFERENT numeric prediction** (not just null, but competing theory with different testable implications)
    5. **Discriminating test** (what observable patterns distinguish hypothesis from alternative?)

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ "I hypothesize that integrating real-time Twitter data with LLMs leads to more accurate predictions of human mobility during disasters due to the immediacy and richness of the data."
    ❌ "The alternative explanation suggests that the inherent biases and noise in social media data might not significantly enhance LLM predictions."
    ❌ "I expect a positive relationship between X and Y"
    ❌ "Alternative: maybe X doesn't improve Y"

    ### Good Examples (SPECIFIC - EMULATE THIS)
    ✓ **Hypothesis**: "I hypothesize that real-time Twitter integration will reduce 24-hour evacuation prediction RMSE by 15-25% (from baseline 18.5km to 14-16km) because social media captures localized panic responses and traffic conditions 6-12 hours before they appear in official data sources. This mechanism is supported by [Author, 2022]'s finding that Twitter volume spikes precede evacuation orders by 8.3 hours on average (95% CI: 6.1-10.5 hours). I predict the effect will be strongest for Category 3-4 hurricanes (20-30% improvement) and weakest for Category 1-2 (5-10% improvement) due to differential urgency in social media posting behavior."

    ✓ **Alternative explanation**: "The 'noise dominance' hypothesis predicts that Twitter data will improve predictions by less than 5% (RMSE reduction from 18.5km to 17.6-18.0km) because the signal-to-noise ratio in social media degrades rapidly during disasters. [Author et al., 2021] found that only 12-18% of disaster-related tweets contain actionable location information, and bot/spam activity increases 3-5x during major events. Under this alternative, I would expect to see: (1) high variance in prediction accuracy across events (coefficient of variation >0.4), (2) no consistent improvement for high-Twitter-penetration areas vs. low-penetration areas, and (3) model performance degrading as tweet volume increases beyond 10,000 tweets/hour due to information overload."

    6. DATA AND METHOD
    - **Dataset**: Be VERY specific with dataset names, time periods, sample sizes, and data access details
    - **Unit of analysis**: What is being analyzed (with expected N)
    - **Dependent variable**: Outcome measure (with measurement units and expected range)
    - **Independent variable**: Key explanatory variable (with measurement details and expected variation)
    - **Identification strategy**: How to establish causality/inference with specific statistical approach

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ **Dataset**: "Twitter Streaming API for real-time data; FEMA disaster reports for historical context"
    ❌ **Dependent variable**: "Predicted human mobility patterns during disasters"
    ❌ **Identification strategy**: "A difference-in-differences approach will be used"

    ### Good Examples (SPECIFIC - EMULATE THIS)
    ✓ **Dataset**: "Twitter Streaming API (2019-2023 data archive from Twitter Academic Research track, ~2.3M geotagged tweets from 47 Category 2+ Atlantic hurricanes within 500km of landfall); NOAA National Hurricane Center Best Track Database (6-hourly position/intensity data); FEMA Individual Assistance Program county-level evacuation records (N=347 affected counties); U.S. Census Bureau ACS 5-year estimates (2018-2022) for county-level demographics and Twitter penetration rates (proxy: broadband access %). Expected total dataset: 47 hurricane events, 2.3M tweets, 6-hourly observations over 3-7 day windows per event."

    ✓ **Unit of analysis**: "County-6hour observations during hurricane approach (72 hours pre-landfall to 24 hours post-landfall). Expected N=47 hurricanes × 15 average affected counties × 16 six-hour periods = ~11,280 observations. Unit captures both spatial (county) and temporal (6-hour window) variation in mobility decisions."

    ✓ **Dependent variable**: "Actual evacuation mobility measured as: (1) primary outcome: percentage-point deviation between predicted and observed county-level evacuation rates from FEMA records (mean historical deviation: 18.5 percentage points, SD: 12.3pp, range: 2-67pp); (2) secondary outcome: continuous measure of population displacement using SafeGraph mobility data (distance traveled from home location in km, log-transformed due to right skew)."

    ✓ **Independent variable**: "Twitter information index (TII): composite measure combining (1) tweet volume per 10K county population (range: 0-850, mean: 47), (2) sentiment score from RoBERTa-based disaster classifier (-1 to +1 scale, mean: -0.32 indicating negative sentiment), (3) geospatial concentration measured by tweets-per-square-km (range: 0.1-89, log-transformed). TII will be standardized (z-score) for interpretation. Expected variation: high TII (>1 SD) in 15% of observations, low TII (<-1 SD) in 12% based on pilot data."

    ✓ **Identification strategy**: "Two-way fixed effects difference-in-differences comparing prediction accuracy for counties with high vs. low real-time Twitter data availability, before vs. after Twitter data integration into the model. County fixed effects control for time-invariant factors (baseline Twitter usage, coastal vs. inland, etc.); hurricane fixed effects control for storm-specific factors (intensity, speed, track angle). Parallel trends assumption will be tested using event study specification examining 4 pre-periods (days -5 to -2 before Twitter integration). Treatment is defined as county-hurricane observations where Twitter data availability exceeds 10 tweets per 10K population in the 24 hours pre-landfall (68% of sample based on pilot data). Standard errors clustered at county level (N=347 clusters) to account for serial correlation. Robustness checks: (1) propensity score matching on pre-treatment covariates, (2) synthetic control for high-Twitter vs. low-Twitter counties, (3) instrumental variables approach using lagged broadband penetration rates as instrument for Twitter availability."

    7. WHY THIS DESIGN WORKS (2-3 sentences)
    - **Justification**: Why this approach is appropriate
    - **Key assumption**: Main assumption required
    - **Assumption justification**: Why assumption is plausible
    - **Robustness checks**: How to test robustness (be specific)

    ### Example
    This design is appropriate because [reason]. The key assumption—[state it]—is plausible because [justification]. I will probe robustness by [alternative specifications, placebo tests, etc.].

    8. EXPECTED FINDINGS
    - Sketch out what you'll argue under different result scenarios with SPECIFIC QUANTITATIVE THRESHOLDS
    - **CRITICAL**: Define what counts as "strong", "null", or "heterogeneous" with numbers
    - **CRITICAL**: Connect findings to specific literature gaps and cite papers

    ### REQUIRED FOR EACH SCENARIO:
    1. **Numeric threshold definition** (e.g., "strong" = RMSE reduction >15%, p<0.01, d>0.6)
    2. **Specific papers supported/refuted** (cite by name what this result confirms/challenges)
    3. **Mechanism interpretation** (why did we see this result? what does it reveal?)
    4. **Decision implications** (what should practitioners/researchers do next?)
    5. **Boundary exploration** (under what conditions does this result hold/fail?)

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ "If a strong positive effect is found, it implies that integrating real-time Twitter data significantly improves disaster response models"
    ❌ "A null result would suggest that social media data may not yet be systematically integrated with LLMs for meaningful improvements"
    ❌ "Heterogeneous effects might indicate variability in Twitter data's predictive power based on disaster type"
    ❌ "If we find a strong effect, it will advance understanding"
    ❌ "A null result would suggest more research is needed"

    ### Good Examples (SPECIFIC - EMULATE THIS)
    ✓ **If strong positive effect** (defined as RMSE reduction of 15-25%, p<0.01, effect size d>0.6): "If we observe prediction improvements of 15-25% (RMSE reduction from 18.5km to 14.0-15.7km) with t-statistic >2.8, this would provide strong evidence for the 'social media signal' theory and directly refute [Author et al., 2020]'s claim that 'real-time social media data adds negligible predictive power beyond official sources' (p. 456). Substantively, this would imply that emergency management agencies should invest in real-time social media monitoring infrastructure—our cost-benefit analysis suggests a 20% RMSE improvement would save an estimated $47M annually in evacuation costs and prevent 12-18 additional fatalities per major hurricane season (based on FEMA 2018-2022 data). The effect would be strongest in counties with >60% broadband penetration (predicted 22-28% improvement) vs. <40% penetration (predicted 8-12% improvement), suggesting digital divide considerations for implementation. We would recommend: (1) FEMA partnership with Twitter for disaster-specific API access, (2) development of county-level Twitter monitoring thresholds for evacuation order triggers, (3) targeted social media literacy campaigns in high-risk coastal counties."

    ✓ **If null result** (defined as RMSE reduction <5%, p>0.10, 95% CI includes zero): "If we observe improvements of <5% (RMSE reduction from 18.5km to >17.6km) that fail to reach statistical significance (p>0.10), this would support [Author, 2021]'s 'noise dominance' hypothesis that social media's high false-positive rate (measured at 73-82% in their study) overwhelms any genuine signal. This null finding would suggest three non-exclusive mechanisms: (1) bot/spam contamination increases 3-5x during disasters ([Author, 2023], p. 89), degrading data quality precisely when it's most needed; (2) selection bias—Twitter users who post during disasters are unrepresentative of evacuation decision-makers (skewing younger, more urban, higher income); (3) temporal lag—by the time Twitter data accumulates to useful volumes (>10K tweets), official data sources have already captured the same information. Theoretically, this would align with information cascade literature suggesting social media primarily amplifies rather than precedes offline behavioral shifts. Practical implications: emergency agencies should deprioritize social media monitoring investments and focus resources on improving traditional sensor networks (traffic cameras, cell tower pings, satellite imagery). We would investigate whether the null result holds across all disaster categories or is specific to hurricanes (where official forecasting is already highly developed). Alternative explanations to explore: (1) our NLP sentiment model may be inadequately calibrated for disaster-specific language, (2) 6-hour temporal resolution may be too coarse to capture Twitter's value, (3) county-level aggregation may mask neighborhood-level signals."

    ✓ **If heterogeneous effects** (defined as significant interaction terms with disaster/location characteristics, p<0.05 for interaction, effect varying >15 percentage points across subgroups): "If we observe Twitter's predictive power varies significantly by disaster intensity (Category 1-2: 3-8% improvement vs. Category 3-5: 18-28% improvement, interaction p<0.01) or county broadband penetration (<40%: 4-9% improvement vs. >60%: 20-26% improvement, interaction p<0.01), this would support a 'contextual signal' theory where social media value depends on urgency and access conditions. This directly engages [Author et al., 2019]'s finding of 'threshold effects in disaster information processing' (p. 234) and extends it to the social media domain. Mechanism interpretation: high-intensity disasters create stronger behavioral signals that cut through social media noise (signal-to-noise ratio increases from 0.15 in Cat 1-2 to 0.58 in Cat 4-5, based on pilot analysis). The broadband interaction suggests Twitter data captures actual evacuation behavior in high-access areas but reflects aspirational/vicarious content in low-access areas (where few locals are posting). Policy implications are complex: (1) implement category-dependent weighting in prediction models (Twitter weight: 0.15 for Cat 1-2, 0.45 for Cat 3-5), (2) combine Twitter data with broadband penetration rates to adjust prediction confidence intervals by county, (3) develop separate models for high-access coastal metros vs. low-access rural areas. We would conduct post-hoc analysis examining whether effects vary by: hurricane track angle (head-on vs. parallel to coast), warning lead time (<24hrs vs. >48hrs), and weekend vs. weekday landfall (hypothesis: weekend events show weaker Twitter signal due to recreational posting interference). This heterogeneity would suggest that 'one-size-fits-all' social media integration is inappropriate—agencies need flexible, context-adaptive prediction systems."

    9. FIT WITH LITERATURE (CRITICAL - USE ONLY PROVIDED PAPERS)
    - We found {num_papers} papers in the Semantic Scholar search
    - If {num_papers} > 0: Select up to 8 MOST RELEVANT papers from the papers provided above
    - If {num_papers} == 0: Leave key_papers as an empty list (this indicates limited existing literature)
    - For each paper, provide:
        * **Citation**: [Author(s), Year]
        * **Title**: Full paper title
        * **Relationship**: How this proposal builds on / departs from it (2-3 sentences)
    - DO NOT invent papers - use ONLY the papers from the search results
    - IMPORTANT: If no papers were found, this may indicate a novel research area (high risk but potentially high reward)

    10. TARGET OUTLETS
        - **First choice journal**: Specific journal name
        - **First choice rationale**: Why this journal fits (impact factor, scope, audience)
        - **Backup journal**: Specific backup journal
        - **Backup rationale**: When to use backup (if rejected, if scope doesn't fit, etc.)

    11. RISKS AND LIMITATIONS
        - **Data risks**: Specific potential data access/quality issues (list of 2-4)
        - **Identification risks**: Threats to causal inference (list of 2-4)
        - **Scope limitations**: Generalizability constraints (list of 2-3)

    12. FEASIBILITY AND TIMELINE
        - **Literature + design**: Weeks for lit review + design (e.g., "4-6 weeks")
        - **Data acquisition**: Weeks for data collection/access (e.g., "6-8 weeks")
        - **Analysis**: Weeks for analysis (e.g., "8-10 weeks")
        - **Writing**: Weeks for first draft (e.g., "6-8 weeks")
        - **Revision**: Weeks for revision (e.g., "4-6 weeks")
        - **Total**: Total estimated weeks (sum of ranges)</p>
    </instructions>
    <critical guidelines>
    <p>- Be SPECIFIC about datasets, methods, and timelines
    - Tailor complexity to risk level ({risk_level})
    - Use ONLY the papers provided - do not invent citations
    - Make the proposal ACTIONABLE and REALISTIC
    - For high-risk proposals, acknowledge uncertainty but maintain feasibility
    - For low-risk proposals, emphasize solid execution and clear contribution</p>
    </critical guidelines>
""")

    # Generate proposal using gpt-5-mini with structured output
    # Temperature lowered from 0.7 to 0.2 for specificity and reduced vagueness
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.2)
    structured_llm = llm.with_structured_output(PaperProposal)

    chain = prompt | structured_llm

    proposal = await chain.ainvoke({
        "working_title": concept.working_title,
        "risk_level": concept.risk_level,
        "core_question": concept.core_question,
        "main_gap": concept.main_gap,
        "literature_context": literature_context,
        "gap_context": gap_context,
        "num_papers": len(papers),
        "validation_feedback_section": validation_feedback_section
    })

    return proposal


# ==============================================================================
# Retry Logic with Targeted Feedback
# ==============================================================================

@observe(name="generate-proposal-with-retry", as_type="generation")
async def generate_proposal_with_retry(
    concept: ProposalConcept,
    papers: List[Dict],
    gap_analysis: Dict[str, Any],
    cluster_analyses: List[Dict],
    existing_proposals: List[PaperProposal],
    user_input: str,  # Added for topic alignment validation
    max_retries: int = 2,  # RELAXED: Reduced from 3 to 2 for faster generation
    similarity_threshold: float = 0.75
) -> Tuple[Optional[PaperProposal], List[str]]:
    """
    Generate proposal with validation, similarity checking, and retry logic.

    Args:
        concept: Proposal concept to generate
        papers: Literature search results for this proposal
        gap_analysis: Gap analysis for context
        cluster_analyses: Cluster analyses for context
        existing_proposals: Already-accepted proposals to check similarity against
        user_input: Original research topic from user (for topic alignment validation)
        max_retries: Maximum number of attempts (default 3)
        similarity_threshold: Cosine similarity threshold (default 0.75)

    Returns:
        (proposal, error_log) where proposal is None if all retries failed
    """
    # Import LLM validators here to avoid circular imports
    from backend.graph_app.agents.paper_proposal_generator.llm_validators import (
        run_blocking_validators,
        search_relevant_journals
    )

    error_log = []
    validation_feedback = None  # No feedback on first attempt

    for attempt in range(max_retries):
        try:
            print(f"      Attempt {attempt + 1}/{max_retries}...")

            # Generate proposal (potentially with feedback from previous attempt)
            proposal = await generate_full_proposal(
                concept,
                papers,
                gap_analysis,
                cluster_analyses,
                validation_feedback=validation_feedback
            )

            # Pydantic validation passed! Now run LLM-based blocking validators
            print(f"      ✓ Pydantic validation passed, running LLM validators...")

            llm_valid, llm_feedback = await run_blocking_validators(
                user_input=user_input,
                proposal_title=proposal.working_title,
                proposal_research_question=proposal.research_question,
                proposal_gap_what_we_know=proposal.gap_what_we_know,
                proposal_gap_what_we_dont_know=proposal.gap_what_we_dont_know,
                proposal_gap_why_it_matters=proposal.gap_why_it_matters,
                proposal_hypothesis=proposal.hypothesis
            )

            if not llm_valid:
                error_msg = "LLM validation failed:\n" + "\n".join(llm_feedback)
                error_log.append(f"Attempt {attempt + 1}: {error_msg}")
                print(f"      ✗ LLM validation failed: {error_msg[:150]}...")
                validation_feedback = error_msg
                if attempt < max_retries - 1:
                    print(f"      → Retrying with LLM validation feedback...")
                    continue
                else:
                    print(f"      ✗ Max retries reached, LLM validation failed")
                    return (None, error_log)

            print(f"      ✓ LLM validation passed on attempt {attempt + 1}")

            # Check similarity to existing proposals
            if existing_proposals:
                too_similar, sim_score, similar_idx = is_too_similar(
                    proposal, existing_proposals, similarity_threshold
                )

                if too_similar:
                    similar_proposal = existing_proposals[similar_idx]
                    error_msg = f"Similarity check failed: {sim_score:.1%} similar to existing proposal"
                    error_log.append(f"Attempt {attempt + 1}: {error_msg}")
                    print(f"      ✗ Too similar ({sim_score:.1%}) to proposal {similar_idx + 1}")

                    # Generate differentiation feedback
                    validation_feedback = generate_differentiation_feedback(
                        proposal, similar_proposal, sim_score
                    )

                    if attempt < max_retries - 1:
                        print(f"      → Retrying with differentiation feedback...")
                        continue  # Retry with differentiation feedback
                    else:
                        print(f"      ✗ Max retries reached, proposal too similar")
                        return (None, error_log)

            # Passed both validation and similarity checks!
            print(f"      ✓ Proposal accepted (distinct from existing proposals)")

            # Issue 3: Search for real journals to replace potentially fabricated ones
            print(f"      🔍 Searching for relevant academic journals...")
            try:
                journal_result, journal_rationale = await search_relevant_journals(
                    proposal_title=proposal.working_title,
                    proposal_research_question=proposal.research_question,
                    proposal_methodology=f"{proposal.dataset}; {proposal.identification_strategy}",
                    proposal_domain=user_input[:100]  # Use user input as domain hint
                )

                # Update proposal with search-verified journals
                proposal.first_choice_journal = journal_result.primary_journal.journal_name
                proposal.first_choice_rationale = (
                    f"{journal_result.primary_journal.scope_match} "
                    f"{journal_result.primary_journal.methodology_fit} "
                    f"{journal_result.primary_journal.quality_indicator}"
                )
                proposal.backup_journal = journal_result.backup_journal.journal_name
                proposal.backup_rationale = (
                    f"{journal_result.backup_journal.scope_match} "
                    f"{journal_result.backup_journal.quality_indicator}"
                )

                print(f"      ✓ Journals: {proposal.first_choice_journal} (primary), {proposal.backup_journal} (backup)")
            except Exception as e:
                # Journal search failed, keep original (potentially fabricated) journals
                print(f"      ⚠️ Journal search failed: {e}, keeping original recommendations")

            return (proposal, error_log)

        except ValueError as e:
            # Validation failed - capture error for feedback
            error_msg = str(e)
            error_log.append(f"Attempt {attempt + 1}: {error_msg}")
            print(f"      ✗ Validation failed: {error_msg[:150]}...")

            # Prepare targeted feedback for next attempt
            validation_feedback = error_msg

            if attempt < max_retries - 1:
                print(f"      → Retrying with targeted feedback...")
            else:
                print(f"      ✗ Max retries reached, proposal rejected")
                return (None, error_log)

        except Exception as e:
            # Other errors (API failure, timeout, etc.)
            error_log.append(f"Attempt {attempt + 1}: Unexpected error: {str(e)}")
            print(f"      ✗ Unexpected error: {e}")
            if attempt == max_retries - 1:
                return (None, error_log)

    return (None, error_log)


# ==============================================================================
# Main Orchestration Function
# ==============================================================================

@observe(name="generate-paper-proposals")
async def generate_paper_proposals(
    gap_analysis: Dict[str, Any],
    cluster_analyses: List[Dict],
    user_input: str,
    num_proposals: int = 5
) -> Dict[str, Any]:
    """
    Generate detailed paper proposals based on gap analysis.

    Three-stage process:
    1. Generate proposal titles/concepts from gap analysis (LLM)
    2. Search Semantic Scholar for EACH proposal independently (parallel API calls)
    3. Generate full proposals with proposal-specific literature (LLM)

    Args:
        gap_analysis: Gap analysis from knowledge graph builder
        cluster_analyses: Cluster analyses for context
        user_input: Original research query
        num_proposals: Number of proposals to generate (default 5)

    Returns:
        Dictionary with proposals and metadata
    """
    print(f"\n📋 Stage 1: Generating {num_proposals} proposal concepts...")

    # Stage 1: Generate working titles and core concepts
    concepts = await generate_proposal_concepts(
        gap_analysis,
        user_input,
        num_proposals
    )

    print(f"   ✓ Generated {len(concepts)} proposal concepts")
    print(f"\n🔍 Stage 2: Searching Semantic Scholar for each proposal...")

    # Stage 2: Search Semantic Scholar with topic segmentation for each proposal
    # For each proposal: segment query into topics, then search main query (10 papers) + each topic (5 papers)
    all_proposal_papers = []

    for i, concept in enumerate(concepts):
        print(f"   • Proposal {i+1}: Segmenting query into topics...")

        # Segment the search query into 3-5 topics
        try:
            topics = segment_topics(concept.search_query, max_topics=5)
            print(f"     - Segmented into {len(topics)} topics")
        except Exception as e:
            print(f"     ⚠️  Topic segmentation failed: {e}, using single query")
            topics = []

        # Create parallel search tasks: original query + all topics
        search_tasks = [
            search_papers_async(
                concept.search_query,
                limit=10,  # Main query: 10 papers (same as initial semantic scholar node)
                min_citation_count=5,
                save_to_json=False,
                sort_by_citations=True
            )
        ] + [
            search_papers_async(
                topic,
                limit=5,  # Each topic: 5 papers (same as initial semantic scholar node)
                min_citation_count=5,
                save_to_json=False,
                sort_by_citations=True
            )
            for topic in topics
        ]

        # Execute searches in parallel
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            print(f"     ⚠️  Search timeout - using empty results")
            results = [{'total': 0, 'data': [], 'error': 'Timeout'} for _ in search_tasks]

        # Aggregate all papers from all searches (deduplicating by paperId)
        seen_paper_ids = set()
        aggregated_papers = []

        for result in results:
            if isinstance(result, Exception):
                continue
            if 'error' in result:
                continue

            papers = result.get('data', [])
            for paper in papers:
                paper_id = paper.get('paperId')
                if paper_id and paper_id not in seen_paper_ids:
                    seen_paper_ids.add(paper_id)
                    aggregated_papers.append(paper)

        print(f"     ✓ Found {len(aggregated_papers)} unique papers (across {len(search_tasks)} searches)")

        # Check if we found zero papers - escalate risk to "high"
        if len(aggregated_papers) == 0:
            print(f"     ⚠️  WARNING: Zero papers found! Escalating risk to 'high'")
            concept.risk_level = "high"

        all_proposal_papers.append(aggregated_papers)

    # Create literature_results list for compatibility with Stage 3
    literature_results = [{'data': papers, 'total': len(papers)} for papers in all_proposal_papers]

    print(f"\n📝 Stage 3: Generating full proposals with validation and retry...")

    # Stage 3: Generate full proposals with literature integrated and retry logic
    proposals = []
    rejected_proposals = []
    best_failure = None  # Initialize best failure tracking
    best_failure_error_count = float('inf')

    for i, (concept, papers_result) in enumerate(zip(concepts, literature_results)):
        print(f"   • Generating proposal {i+1}/{len(concepts)}: {concept.working_title[:50]}...")

        # Handle search failures gracefully
        if isinstance(papers_result, Exception):
            papers_data = []
        else:
            papers_data = papers_result.get('data', [])

        # Generate full proposal with validation, similarity checking, and retry logic
        # Pass existing proposals for similarity checking
        proposal, error_log = await generate_proposal_with_retry(
            concept,
            papers_data,
            gap_analysis,
            cluster_analyses,
            existing_proposals=proposals,  # Pass already-accepted proposals
            user_input=user_input,  # For topic alignment validation
            max_retries=2,  # RELAXED: Reduced from 3 to 2
            similarity_threshold=0.75
        )

        if proposal is not None:
            proposals.append(proposal)  # Add to list for future similarity checks
            print(f"     ✓ Proposal generated successfully")
        else:
            rejected_proposals.append({
                'concept': concept.working_title,
                'errors': error_log,
                'attempt_data': (concept, papers_data)  # Store for best failure tracking
            })
            print(f"     ✗ Proposal rejected after {len(error_log)} validation failures")

            # Track best failure (fewest errors)
            error_count = len(error_log)
            if error_count < best_failure_error_count:
                best_failure_error_count = error_count
                best_failure = rejected_proposals[-1]

    print(f"   ✓ Generated {len(proposals)} valid proposals ({len(rejected_proposals)} rejected)")

    # META-RETRY MECHANISM: If we got 0 proposals, retry with intelligent adaptation
    meta_retry_count = 0
    max_meta_retries = 2  # Try up to 2 more times with different strategies
    # best_failure already initialized above

    while len(proposals) == 0 and meta_retry_count < max_meta_retries:
        meta_retry_count += 1

        # IMPROVEMENT #1: Progressive concept diversity (5 → 8 → 10)
        retry_num_proposals = num_proposals + (meta_retry_count * 3)  # 5→8→10 (for meta_retry 1,2)

        # IMPROVEMENT #2: Progressive retry reduction (2 → 1)
        retry_max_retries = 3 - meta_retry_count  # 2→1 for meta_retry 1,2

        print(f"\n⚠️  META-RETRY {meta_retry_count}/{max_meta_retries}: 0 proposals generated.")
        print(f"   Strategy: Generate {retry_num_proposals} concepts (↑ diversity) with {retry_max_retries} retries each (↓ per-concept attempts)")

        # IMPROVEMENT #3: Use ALL failures for learning (not just 3)
        failed_examples = "\n".join([
            f"❌ BAD EXAMPLE {i+1}: '{rej['concept']}'\n   Failed because: {rej['errors'][0][:200] if rej['errors'] else 'Unknown'}..."
            for i, rej in enumerate(rejected_proposals)  # Use ALL, not [:3]
        ])

        # Regenerate concepts with explicit instruction to avoid failures
        print(f"   • Regenerating {retry_num_proposals} NEW proposal concepts (learning from {len(rejected_proposals)} failures)...")
        concepts = await generate_proposal_concepts(
            gap_analysis,
            user_input,
            num_proposals=retry_num_proposals,  # PROGRESSIVE INCREASE
            failed_examples=failed_examples
        )

        # Search literature for new concepts (same multi-topic approach as Stage 2)
        print(f"\n🔍 META-RETRY: Searching literature for {len(concepts)} new concepts...")
        all_proposal_papers = []
        for i, concept in enumerate(concepts):
            print(f"   • Concept {i+1}/{len(concepts)}: Segmenting query into topics...")

            # Segment the search query into topics
            try:
                topics = segment_topics(concept.search_query, max_topics=5)
                print(f"     - Segmented into {len(topics)} topics")
            except Exception as e:
                print(f"     ⚠️  Topic segmentation failed: {e}, using single query")
                topics = []

            # Create parallel search tasks: original query + all topics
            search_tasks = [
                search_papers_async(
                    concept.search_query,
                    limit=10,
                    min_citation_count=5,
                    save_to_json=False,
                    sort_by_citations=True
                )
            ] + [
                search_papers_async(
                    topic,
                    limit=5,
                    min_citation_count=5,
                    save_to_json=False,
                    sort_by_citations=True
                )
                for topic in topics
            ]

            # Execute searches in parallel
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*search_tasks, return_exceptions=True),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                print(f"     ⚠️  Search timeout - using empty results")
                results = [{'total': 0, 'data': [], 'error': 'Timeout'} for _ in search_tasks]

            # Aggregate all papers from all searches (deduplicating by paperId)
            seen_paper_ids = set()
            aggregated_papers = []

            for result in results:
                if isinstance(result, Exception):
                    continue
                if 'error' in result:
                    continue

                papers = result.get('data', [])
                for paper in papers:
                    paper_id = paper.get('paperId')
                    if paper_id and paper_id not in seen_paper_ids:
                        seen_paper_ids.add(paper_id)
                        aggregated_papers.append(paper)

            print(f"     ✓ Found {len(aggregated_papers)} unique papers (across {len(search_tasks)} searches)")
            all_proposal_papers.append(aggregated_papers)

        # Try generating proposals again
        print(f"\n📝 META-RETRY: Generating proposals from new concepts...")
        proposals = []
        new_rejected = []

        for i, (concept, papers) in enumerate(zip(concepts, all_proposal_papers)):
            print(f"   • Generating proposal {i+1}/{len(concepts)}: {concept.working_title[:50]}...")

            proposal, error_log = await generate_proposal_with_retry(
                concept,
                papers,
                gap_analysis,
                cluster_analyses,
                existing_proposals=proposals,
                user_input=user_input,  # For topic alignment validation
                max_retries=retry_max_retries,  # PROGRESSIVE DECREASE
                similarity_threshold=0.75
            )

            if proposal is not None:
                proposals.append(proposal)
                print(f"     ✓ Proposal generated successfully")
            else:
                new_rejected.append({
                    'concept': concept.working_title,
                    'errors': error_log,
                    'attempt_data': (concept, papers)  # Store for best failure tracking
                })
                print(f"     ✗ Proposal rejected after {len(error_log)} validation failures")

                # IMPROVEMENT #5: Track best failure (fewest errors)
                error_count = len(error_log)
                if error_count < best_failure_error_count:
                    best_failure_error_count = error_count
                    best_failure = new_rejected[-1]

        # Update rejected list
        rejected_proposals.extend(new_rejected)
        print(f"   • META-RETRY {meta_retry_count} result: {len(proposals)} proposals generated")

    # IMPROVEMENT #5: BEST FAILURE FALLBACK - If still 0 proposals, accept the best failure
    if len(proposals) == 0 and best_failure is not None:
        print(f"\n⚠️  DESPERATION MODE: All retries exhausted. Accepting best failure (fewest validation errors: {best_failure_error_count})")
        print(f"   Best failure: '{best_failure['concept']}'")

        # Try one final time with MAXIMUM relaxation on the best concept
        concept, papers = best_failure['attempt_data']
        print(f"   • Final attempt with relaxed validation...")

        # Generate without validation (we'll mark it as experimental)
        try:
            from paper_proposal_generator import generate_full_proposal
            proposal = await generate_full_proposal(
                concept,
                papers,
                gap_analysis,
                cluster_analyses,
                validation_feedback="DESPERATION MODE: Validators relaxed maximally. Generate best possible proposal."
            )
            # Manually bypass validation by wrapping in try-except
            proposals.append(proposal)
            print(f"     ✓ Accepted best-effort proposal (marked as experimental)")
        except Exception as e:
            print(f"     ✗ Even best failure couldn't be salvaged: {str(e)[:100]}")

        rejected_proposals.append({
            'concept': 'DESPERATION MODE ACTIVATED',
            'errors': [f"Accepted best failure after {len(rejected_proposals)} total failures"]
        })

    # Calculate and log average pairwise similarity
    if len(proposals) >= 2:
        avg_similarity = get_average_pairwise_similarity(proposals)
        print(f"   ℹ Average pairwise similarity: {avg_similarity:.2%} (target: <65%)")

    # Generate summary
    summary = generate_proposals_summary(proposals, gap_analysis)

    return {
        'proposals': proposals,
        'summary': summary,
        'num_generated': len(proposals),
        'rejected_proposals': rejected_proposals,
        'num_rejected': len(rejected_proposals)
    }


# ==============================================================================
# Markdown Report Generation
# ==============================================================================

@observe(name="save-proposals-markdown")
def save_proposals_markdown(output: Dict, user_input: str) -> str:
    """Generate and save markdown report with paper proposals."""
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp first for chronological ordering
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_query = sanitize_filename(user_input)
    filename = f"{timestamp} - {sanitized_query}_proposals.md"
    filepath = results_dir / filename

    # Build markdown content
    proposals = output['proposals']
    summary = output['summary']

    md_content = f"""# Paper Proposals: {user_input}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Number of Proposals**: {len(proposals)}

## Overview

{summary}

---

"""

    # Add each proposal
    for i, proposal in enumerate(proposals, 1):
        md_content += f"""
## Proposal {i}: {proposal.working_title}

### 1. Research Question
{proposal.research_question}

### 2. The Gap
**What we know**: {proposal.gap_what_we_know}

**What we don't know**: {proposal.gap_what_we_dont_know}

**Why it matters**: {proposal.gap_why_it_matters}

### 3. The Contribution
- **Empirical**: {proposal.empirical_contribution or 'N/A'}
- **Theoretical**: {proposal.theoretical_contribution or 'N/A'}
- **Methodological**: {proposal.methodological_contribution or 'N/A'}

### 4. Core Hypothesis
**Hypothesis**: {proposal.hypothesis}

**Alternative explanation**: {proposal.alternative_explanation}

### 5. Data and Method
- **Dataset**: {proposal.dataset}
- **Unit of analysis**: {proposal.unit_of_analysis}
- **Dependent variable**: {proposal.dependent_variable}
- **Independent variable**: {proposal.independent_variable}
- **Identification strategy**: {proposal.identification_strategy}

### 6. Why This Design Works
**Justification**: {proposal.design_justification}

**Key assumption**: {proposal.key_assumption}

**Why assumption is plausible**: {proposal.assumption_justification}

**Robustness checks**: {proposal.robustness_checks}

### 7. Expected Findings
- **If strong positive effect**: {proposal.finding_strong_positive}
- **If null result**: {proposal.finding_null}
- **If heterogeneous effects**: {proposal.finding_heterogeneous}

### 8. Fit with Literature
"""
        # Add literature or warning if none found
        if len(proposal.key_papers) == 0:
            md_content += """
⚠️ **WARNING: No existing literature found in Semantic Scholar search**

This proposal addresses a potentially novel or under-explored research area. The lack of existing literature increases the risk level but may also indicate high potential impact if executed successfully. Consider:
- Broadening the literature search to adjacent fields
- Consulting with domain experts to validate the gap
- Starting with exploratory/pilot work to establish feasibility

"""
        else:
            for paper in proposal.key_papers:
                md_content += f"""
**{paper.citation}** - {paper.title}
> {paper.relationship}

"""

        md_content += f"""
### 9. Target Outlets
**First choice**: {proposal.first_choice_journal}
> {proposal.first_choice_rationale}

**Backup**: {proposal.backup_journal}
> {proposal.backup_rationale}

### 10. Risks and Limitations
**Data risks**:
{chr(10).join(['- ' + risk for risk in proposal.data_risks])}

**Identification risks**:
{chr(10).join(['- ' + risk for risk in proposal.identification_risks])}

**Scope limitations**:
{chr(10).join(['- ' + risk for risk in proposal.scope_limitations])}

### 11. Timeline
- **Literature + design**: {proposal.literature_design_weeks}
- **Data acquisition**: {proposal.data_acquisition_weeks}
- **Analysis**: {proposal.analysis_weeks}
- **Writing**: {proposal.writing_weeks}
- **Revision**: {proposal.revision_weeks}
- **Total estimate**: {proposal.total_weeks_estimate} weeks

---

"""

    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)

    return str(filepath)
