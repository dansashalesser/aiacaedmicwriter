"""Paper Proposal Generator - Generates detailed academic paper proposals from gap analysis."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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
    literature_design_weeks: str = Field(description="Weeks for lit review + design (e.g., '1-4')")
    data_acquisition_weeks: str = Field(description="Weeks for data (e.g., '5-8')")
    analysis_weeks: str = Field(description="Weeks for analysis (e.g., '9-14')")
    writing_weeks: str = Field(description="Weeks for draft (e.g., '15-18')")
    revision_weeks: str = Field(description="Weeks for revision (e.g., '19-22')")
    total_weeks_estimate: int = Field(description="Total estimated weeks")


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

    summary = f"""This research agenda presents {len(proposals)} paper proposals addressing key gaps identified in the literature:

Key Gaps Addressed:
{chr(10).join(['- ' + gap for gap in gap_analysis.get('research_gaps', [])[:3]])}

Proposed Papers:
{chr(10).join([f"{i+1}. {title}" for i, title in enumerate(titles)])}

These proposals range from high-risk, high-impact theoretical contributions to safer empirical extensions, providing a balanced research portfolio.
"""

    return summary


# ==============================================================================
# Stage 1: Generate Proposal Concepts
# ==============================================================================

@observe(name="generate-proposal-concepts", as_type="generation")
async def generate_proposal_concepts(
    gap_analysis: Dict[str, Any],
    user_input: str,
    num_proposals: int = 5
) -> List[ProposalConcept]:
    """
    Generate initial proposal concepts with risk stratification.

    Returns concepts optimized for Semantic Scholar searching.
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
    {priority_recommendations}</p>
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

    # Generate concepts using gpt-4o with structured output
    llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
    structured_llm = llm.with_structured_output(ProposalConcepts)

    chain = prompt | structured_llm

    result = await chain.ainvoke({
        "user_input": user_input,
        "research_gaps": research_gaps,
        "untried_directions": untried_directions,
        "priority_recommendations": priority_recommendations,
        "num_proposals": num_proposals,
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
    cluster_analyses: List[Dict]
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

    prompt = ChatPromptTemplate.from_template("""# Paper Ideation Generator
    <directives>
    <general directives>
    <p>Make sure you are correct, and think about your response step by step. This is VERY important to me!</p>
    </general directives>
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
    - Why doesn't existing work answer this question? Be specific about what's missing—not just "no one has studied this."
    - **What we know**: Summarize what the literature establishes WITH SPECIFIC CITATIONS
    - **What we don't know**: The specific intellectual gap WITH QUANTIFICATION
    - **Why it matters**: Theoretical/practical importance WITH CONCRETE IMPACT METRICS

    ### Bad Examples for "Why It Matters" (TOO VAGUE - DO NOT EMULATE)
    ❌ "Enhancing predictive models with real-time data could significantly improve disaster response efforts, potentially saving lives and resources"
    ❌ "Understanding this relationship can significantly improve disaster preparedness"
    ❌ "This is important for advancing our theoretical understanding"

    ### Good Examples for "Why It Matters" (SPECIFIC - EMULATE THIS)
    ✓ "Current FEMA evacuation models have a mean absolute error of 18.5 percentage points in predicting county-level evacuation rates ([Author, 2022], p. 234), leading to systematic over-evacuation in low-risk areas (costing $12-18M per false alarm) and under-evacuation in high-risk areas (contributing to an estimated 23% of hurricane-related fatalities in 2017-2022). Reducing prediction error by 15-25% would enable more targeted evacuation orders, potentially saving $47M annually in unnecessary evacuation costs (based on average of 8.3 major hurricanes/year requiring evacuation orders for 2.1M people at $450/person evacuation cost) while preventing an estimated 12-18 additional fatalities per major hurricane season through better resource allocation to truly high-risk areas. Theoretically, this addresses the longstanding debate in disaster sociology between 'information deficit' models (which assume more data improves decisions) and 'information overload' models (which predict diminishing returns)—[Author et al., 2020] argue this question remains 'empirically unresolved due to lack of real-time data integration studies' (p. 456)."

    4. THE CONTRIBUTION (2-3 bullet points)
    - What will readers know after reading your paper that they didn't know before? Be concrete.
    - What new finding this will produce- Include **METHODOLOGICAL CONCEPTS, NUMBERS OR STRUCTURES- DO NOT BE VAGUE!**
    - **CRITICAL**: Each contribution MUST reference specific findings from the literature search or gap analysis
    - **CRITICAL**: Include quantitative expectations (e.g., "X% improvement", "N new variables", "K model architectures")

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

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ "I hypothesize that integrating real-time Twitter data with LLMs leads to more accurate predictions of human mobility during disasters due to the immediacy and richness of the data."
    ❌ "The alternative explanation suggests that the inherent biases and noise in social media data might not significantly enhance LLM predictions."

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

    ### Bad Examples (TOO VAGUE - DO NOT EMULATE)
    ❌ "If a strong positive effect is found, it implies that integrating real-time Twitter data significantly improves disaster response models"
    ❌ "A null result would suggest that social media data may not yet be systematically integrated with LLMs for meaningful improvements"
    ❌ "Heterogeneous effects might indicate variability in Twitter data's predictive power based on disaster type"

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

    # Generate proposal using gpt-4o with structured output
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    structured_llm = llm.with_structured_output(PaperProposal)

    chain = prompt | structured_llm

    proposal = await chain.ainvoke({
        "working_title": concept.working_title,
        "risk_level": concept.risk_level,
        "core_question": concept.core_question,
        "main_gap": concept.main_gap,
        "literature_context": literature_context,
        "gap_context": gap_context,
        "num_papers": len(papers)
    })

    return proposal


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

    print(f"\n📝 Stage 3: Generating full proposals with literature...")

    # Stage 3: Generate full proposals with literature integrated
    proposals = []
    for i, (concept, papers_result) in enumerate(zip(concepts, literature_results)):
        print(f"   • Generating proposal {i+1}/{len(concepts)}: {concept.working_title[:50]}...")

        # Handle search failures gracefully
        if isinstance(papers_result, Exception):
            papers_data = []
        else:
            papers_data = papers_result.get('data', [])

        # Generate full proposal with literature
        proposal = await generate_full_proposal(
            concept,
            papers_data,
            gap_analysis,
            cluster_analyses
        )
        proposals.append(proposal)

    print(f"   ✓ Generated {len(proposals)} complete proposals")

    # Generate summary
    summary = generate_proposals_summary(proposals, gap_analysis)

    return {
        'proposals': proposals,
        'summary': summary,
        'num_generated': len(proposals)
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

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_query = sanitize_filename(user_input)
    filename = f"{sanitized_query}_{timestamp}_proposals.md"
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
