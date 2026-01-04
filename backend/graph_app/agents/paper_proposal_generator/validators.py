"""Validators for paper proposals - enforce anti-vagueness requirements."""

from typing import List, Tuple
import re

# =============================================================================
# Banned Phrase Detection
# =============================================================================

BANNED_PHRASES = [
    "will advance understanding",
    "will present new evidence",
    "will introduce novel approach",
    "will enhance",
    "will provide insights",
    "will advance the field",
    "significantly improve",  # unless followed by number
    "comprehensive dataset",  # unless followed by specifics
    "due to the richness",
    "due to the complexity",
    "due to the importance",
    "real-time data",  # unless followed by specific source
    "novel approach",  # unless followed by specific description
]

VAGUE_PATTERNS = [
    r"will\s+(provide|present|introduce|advance|enhance|improve)\s+(?!.*\d)",  # Future tense without numbers
    r"significant(?!ly.*\d)",  # "significant" without quantification
    r"comprehensive\s+\w+(?!\s+\()",  # "comprehensive X" without specifics in parens
]


def detect_banned_phrases(text: str) -> List[str]:
    """Return list of banned phrases found in text."""
    found = []
    text_lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in text_lower:
            found.append(phrase)
    for pattern in VAGUE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found.append(f"Pattern: {pattern} (matched: {', '.join(matches[:2])})")
    return found


# =============================================================================
# Quantification Checks
# =============================================================================

def contains_numbers(text: str) -> bool:
    """Check if text contains numeric information."""
    # Look for: percentages, decimals, ranges, counts, years
    numeric_patterns = [
        r'\d+\.?\d*%',  # Percentages
        r'\d+\.?\d*',   # Numbers
        r'\d+-\d+',     # Ranges
        r'N\s*=\s*\d+', # Sample sizes
        r'p\s*[<>=]\s*0?\.\d+',  # P-values
        r'\d{4}',       # Years
    ]
    return any(re.search(pattern, text) for pattern in numeric_patterns)


def count_numbers(text: str) -> int:
    """Count numeric mentions in text."""
    return len(re.findall(r'\d+\.?\d*', text))


# =============================================================================
# Citation Checks
# =============================================================================

def contains_citations(text: str) -> bool:
    """Check if text contains citations."""
    citation_patterns = [
        r'\[[\w\s]+,?\s*\d{4}\]',  # [Author, 2023] or [Author et al., 2023]
        r'\([\w\s]+,?\s*\d{4}\)',  # (Author, 2023)
        r'[\w\s]+\s+\(\d{4}\)',    # Author (2023)
        r'et al\.',                 # et al.
    ]
    return any(re.search(pattern, text) for pattern in citation_patterns)


# =============================================================================
# Hypothesis Validation
# =============================================================================

def validate_hypothesis(hypothesis: str, alternative: str) -> Tuple[bool, List[str]]:
    """
    Validate hypothesis contains required components.
    Returns (is_valid, error_messages).
    """
    errors = []

    # Check hypothesis has numbers
    if not contains_numbers(hypothesis):
        errors.append("Hypothesis lacks quantitative predictions (no numbers found)")

    # Check hypothesis length (too short = not specific) - RELAXED to 20 words
    if len(hypothesis.split()) < 20:
        errors.append(f"Hypothesis too short ({len(hypothesis.split())} words, need >20 for specificity)")

    # Check alternative is different from hypothesis - RELAXED to 10 words
    if len(alternative.split()) < 10:
        errors.append(f"Alternative too short ({len(alternative.split())} words, need >10)")

    if not contains_numbers(alternative):
        errors.append("Alternative lacks quantitative predictions")

    # Check alternatives aren't just negations
    negation_indicators = ["will not", "won't", "no effect", "null", "doesn't"]
    if any(phrase in alternative.lower() for phrase in negation_indicators):
        if count_numbers(alternative) < 2:
            errors.append("Alternative appears to be mere negation, not competing theory with different predictions")

    return (len(errors) == 0, errors)


# =============================================================================
# Gap Validation
# =============================================================================

COVERAGE_GAP_INDICATORS = [
    "no one has",
    "no study has",
    "no research has",
    "has not been",
    "have not been",
    "remains unexplored",
    "remains understudied",
    "no prior work",
    "first to",
    "novel combination",
    "unexplored combination",
    "limited research on",
    "few studies have",
    "under-explored",
    "under-studied",
]


def is_coverage_gap(gap_text: str) -> bool:
    """Detect if gap is coverage-based rather than intellectual."""
    gap_lower = gap_text.lower()
    return any(indicator in gap_lower for indicator in COVERAGE_GAP_INDICATORS)


def validate_gap(what_we_know: str, what_we_dont_know: str, why_it_matters: str) -> Tuple[bool, List[str]]:
    """Validate gap is intellectual, not coverage-based."""
    errors = []

    # Check "what we know" has citations
    if not contains_citations(what_we_know):
        errors.append("'What we know' lacks citations to literature")

    # Check "what we know" has numbers
    if not contains_numbers(what_we_know):
        errors.append("'What we know' lacks quantitative findings from literature")

    # Check "what we don't know" isn't coverage gap
    if is_coverage_gap(what_we_dont_know):
        errors.append("Gap appears to be coverage-based ('no one has X+Y') rather than intellectual (assumption Z untested)")

    # Check "what we don't know" identifies assumption/contradiction - RELAXED significantly
    # Expanded to include more research-related terms
    intellectual_indicators = ["assume", "assumption", "contradict", "unclear", "untested", "unresolved", "don't know whether", "don't understand", "unknown", "question", "gap", "lack", "limited", "insufficient", "not", "no", "whether", "how", "what", "why", "explore", "investigate", "examine"]
    if not any(ind in what_we_dont_know.lower() for ind in intellectual_indicators):
        errors.append("'What we don't know' should identify untested assumption or unresolved contradiction")

    # Check "why it matters" has quantification
    if not contains_numbers(why_it_matters):
        errors.append("'Why it matters' lacks quantified impact (needs numbers)")

    return (len(errors) == 0, errors)


# =============================================================================
# Contribution Validation
# =============================================================================

def validate_contribution(contribution: str, contribution_type: str) -> Tuple[bool, List[str]]:
    """Validate contribution is specific, not boilerplate."""
    errors = []

    if not contribution:  # Optional contributions can be None
        return (True, [])

    # Check for numbers - ONLY for empirical, not theoretical/methodological
    if contribution_type == "Empirical" and not contains_numbers(contribution):
        errors.append(f"{contribution_type} contribution lacks quantification (no numbers/percentages/metrics)")

    # Check length (too short = likely boilerplate) - RELAXED to 20 words
    if len(contribution.split()) < 20:
        errors.append(f"{contribution_type} contribution too short ({len(contribution.split())} words, need >20 for specificity)")

    return (len(errors) == 0, errors)


# =============================================================================
# Variable Operationalization Check
# =============================================================================

def validate_variable_operationalization(
    dependent_var: str,
    independent_var: str
) -> Tuple[bool, List[str]]:
    """Check variables include measurement details."""
    errors = []

    # Look for measurement indicators
    measurement_indicators = ["measured as", "measured by", "operationalized", "calculated", "scored", "range", "scale", "proxy"]

    if not any(ind in dependent_var.lower() for ind in measurement_indicators):
        errors.append("Dependent variable lacks operationalization (how is it measured?)")

    if not any(ind in independent_var.lower() for ind in measurement_indicators):
        errors.append("Independent variable lacks operationalization (how is it measured?)")

    # Check for numbers (ranges, scales, etc.)
    if not contains_numbers(dependent_var):
        errors.append("Dependent variable should specify expected range/values")

    if not contains_numbers(independent_var):
        errors.append("Independent variable should specify expected range/values")

    return (len(errors) == 0, errors)


# =============================================================================
# Research Question Validation
# =============================================================================

def validate_research_question(question: str) -> Tuple[bool, List[str]]:
    """Validate research question is specific and answerable."""
    errors = []

    # Check length (too short = vague)
    if len(question.split()) < 15:
        errors.append(f"Research question too short ({len(question.split())} words) - needs specific variables/context")

    # Check for vague language
    vague_terms = ["how can", "how might", "is it possible", "could we", "can we improve"]
    if any(term in question.lower() for term in vague_terms):
        errors.append("Research question uses vague language - should ask 'what/how/why' about specific relationships")

    # Check for quantification or specificity - RELAXED to allow more questions through
    if not contains_numbers(question):
        # If no numbers, should at least have specific constructs or key research terms
        specific_terms = ["effect", "relationship", "impact", "association", "difference", "predict", "influence", "determine", "explain", "measure", "assess", "evaluate", "compare", "analyze", "examine"]
        if not any(term in question.lower() for term in specific_terms):
            errors.append("Research question lacks specificity - should mention specific effects, relationships, or quantities")

    # Check it's actually a question
    if "?" not in question:
        errors.append("Research question should be phrased as a question (ending with ?)")

    return (len(errors) == 0, errors)


# =============================================================================
# Risks and Limitations Validation
# =============================================================================

GENERIC_RISK_PHRASES = [
    "data quality issues",
    "data availability",
    "data access",
    "sample size",
    "limited generalizability",
    "confounding variables",
    "measurement error",
    "selection bias",
    "external validity",
]


def is_generic_risk(risk_text: str) -> bool:
    """Detect if risk is too generic."""
    risk_lower = risk_text.lower()
    # Generic if it's just a phrase from the list without specifics
    if len(risk_text.split()) < 10:  # Too short to be specific
        return any(phrase in risk_lower for phrase in GENERIC_RISK_PHRASES)
    return False


def validate_risks_and_limitations(
    data_risks: List[str],
    identification_risks: List[str],
    scope_limitations: List[str]
) -> Tuple[bool, List[str]]:
    """Validate risks are study-specific, not generic."""
    errors = []

    # Check data risks
    for risk in data_risks:
        if is_generic_risk(risk):
            errors.append(f"Data risk too generic: '{risk[:50]}...' - needs study-specific details")
        if not any(char.isdigit() for char in risk) and len(risk.split()) < 12:
            errors.append(f"Data risk lacks specificity: '{risk[:50]}...' - should include details about specific data sources or quantities")

    # Check identification risks - RELAXED significantly
    for risk in identification_risks:
        if is_generic_risk(risk):
            errors.append(f"Identification risk too generic: '{risk[:50]}...' - needs specific threat to THIS study's design")
        # Identification risks should mention design elements - RELAXED with more terms
        design_terms = ["assumption", "control", "fixed effects", "instrument", "parallel trends", "selection", "endogeneity", "bias", "confound", "causal", "validity", "reliability", "measurement", "data", "sample", "model", "variable", "factor", "effect"]
        if not any(term in risk.lower() for term in design_terms):
            errors.append(f"Identification risk should reference specific design elements: '{risk[:50]}...'")

    # Check scope limitations - RELAXED, only check if very short
    for limitation in scope_limitations:
        if len(limitation.split()) < 5:  # Only reject if extremely short
            errors.append(f"Scope limitation too vague: '{limitation[:50]}...' - needs more detail")

    # Check minimum counts (should have multiple risks in each category)
    if len(data_risks) < 2:
        errors.append(f"Need at least 2 data risks (found {len(data_risks)})")
    if len(identification_risks) < 2:
        errors.append(f"Need at least 2 identification risks (found {len(identification_risks)})")
    if len(scope_limitations) < 2:
        errors.append(f"Need at least 2 scope limitations (found {len(scope_limitations)})")

    return (len(errors) == 0, errors)


# =============================================================================
# Timeline Validation
# =============================================================================

def parse_weeks_range(weeks_str: str) -> Tuple[int, int]:
    """Parse week range string like '4-6' into (min, max)."""
    try:
        if '-' in weeks_str:
            parts = weeks_str.split('-')
            return (int(parts[0]), int(parts[1]))
        else:
            # Single number
            val = int(weeks_str)
            return (val, val)
    except (ValueError, IndexError):
        return (0, 0)


def validate_timeline(
    literature_design_weeks: str,
    data_acquisition_weeks: str,
    analysis_weeks: str,
    writing_weeks: str,
    revision_weeks: str,
    total_weeks_estimate: int
) -> Tuple[bool, List[str]]:
    """Validate timeline is realistic and internally consistent."""
    errors = []

    # Parse all ranges
    lit_min, lit_max = parse_weeks_range(literature_design_weeks)
    data_min, data_max = parse_weeks_range(data_acquisition_weeks)
    analysis_min, analysis_max = parse_weeks_range(analysis_weeks)
    writing_min, writing_max = parse_weeks_range(writing_weeks)
    revision_min, revision_max = parse_weeks_range(revision_weeks)

    # Check that ranges were parsed successfully
    if lit_min == 0 or data_min == 0 or analysis_min == 0:
        errors.append("Timeline ranges must be valid (e.g., '4-6' or '8')")
        return (False, errors)

    # Calculate min and max totals
    min_total = lit_min + data_min + analysis_min + writing_min + revision_min
    max_total = lit_max + data_max + analysis_max + writing_max + revision_max

    # Check total is within bounds
    if total_weeks_estimate < min_total:
        errors.append(f"Total weeks ({total_weeks_estimate}) less than sum of minimums ({min_total})")
    if total_weeks_estimate > max_total + 4:  # Allow small buffer for overlap
        errors.append(f"Total weeks ({total_weeks_estimate}) exceeds sum of maximums ({max_total})")

    # Check minimum realistic duration (academic papers take time)
    if total_weeks_estimate < 12:
        errors.append(f"Timeline too short ({total_weeks_estimate} weeks) - academic papers typically require 3+ months (12+ weeks)")

    # Check maximum realistic duration (shouldn't be multi-year for single paper)
    if total_weeks_estimate > 104:  # 2 years
        errors.append(f"Timeline too long ({total_weeks_estimate} weeks = {total_weeks_estimate/52:.1f} years) - should be completable within 1-2 years")

    # Check phase durations are realistic
    if lit_max < 2:
        errors.append("Literature review too short - need at least 2 weeks")
    if data_max < 4:
        errors.append("Data acquisition too short - need at least 4 weeks for realistic data collection")
    if analysis_max < 4:
        errors.append("Analysis too short - need at least 4 weeks for thorough analysis")
    if writing_max < 4:
        errors.append("Writing too short - need at least 4 weeks for first draft")

    return (len(errors) == 0, errors)


# =============================================================================
# Outlet Validation
# =============================================================================

VAGUE_JOURNAL_TERMS = [
    "top journal",
    "leading journal",
    "major journal",
    "prestigious journal",
    "high-impact journal",
    "tier 1 journal",
    "A* journal",
]


def validate_outlet(
    journal: str,
    rationale: str,
    outlet_type: str
) -> Tuple[bool, List[str]]:
    """Validate outlet is specific with quantified rationale."""
    errors = []

    # Check journal name is specific
    journal_lower = journal.lower()
    if any(vague in journal_lower for vague in VAGUE_JOURNAL_TERMS):
        errors.append(f"{outlet_type} journal too vague: '{journal}' - provide specific journal name")

    # Check rationale has quantification
    if not contains_numbers(rationale):
        errors.append(f"{outlet_type} rationale lacks quantification (e.g., impact factor, acceptance rate, similar papers published)")

    # Check rationale length (too short = not substantive)
    if len(rationale.split()) < 15:
        errors.append(f"{outlet_type} rationale too short ({len(rationale.split())} words) - need specific justification")

    return (len(errors) == 0, errors)
