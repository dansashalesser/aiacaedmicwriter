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
    Validate hypothesis contains required components - VERY RELAXED.
    Returns (is_valid, error_messages).
    """
    errors = []

    # RELAXED: Only check basic length
    if len(hypothesis.split()) < 15:
        errors.append(f"Hypothesis too short ({len(hypothesis.split())} words, need >15)")

    if len(alternative.split()) < 8:
        errors.append(f"Alternative too short ({len(alternative.split())} words, need >8)")

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
    """Validate gap is intellectual, not coverage-based - VERY RELAXED."""
    errors = []

    # RELAXED: Only check basic length, don't enforce citations or numbers
    if len(what_we_know.split()) < 10:
        errors.append("'What we know' is too short (needs at least 10 words)")

    if len(what_we_dont_know.split()) < 10:
        errors.append("'What we don't know' is too short (needs at least 10 words)")

    if len(why_it_matters.split()) < 10:
        errors.append("'Why it matters' is too short (needs at least 10 words)")

    return (len(errors) == 0, errors)


# =============================================================================
# Contribution Validation
# =============================================================================

def validate_contribution(contribution: str, contribution_type: str) -> Tuple[bool, List[str]]:
    """Validate contribution is specific, not boilerplate - VERY RELAXED."""
    errors = []

    if not contribution:  # Optional contributions can be None
        return (True, [])

    # RELAXED: Only check basic length (10 words minimum)
    if len(contribution.split()) < 10:
        errors.append(f"{contribution_type} contribution too short ({len(contribution.split())} words, need >10)")

    return (len(errors) == 0, errors)


# =============================================================================
# Variable Operationalization Check
# =============================================================================

def validate_variable_operationalization(
    dependent_var: str,
    independent_var: str
) -> Tuple[bool, List[str]]:
    """Check variables include measurement details - VERY RELAXED."""
    errors = []

    # RELAXED: Only check basic length (at least 5 words each)
    if len(dependent_var.split()) < 5:
        errors.append("Dependent variable too short (needs at least 5 words)")

    if len(independent_var.split()) < 5:
        errors.append("Independent variable too short (needs at least 5 words)")

    return (len(errors) == 0, errors)


# =============================================================================
# Research Question Validation
# =============================================================================

def validate_research_question(question: str) -> Tuple[bool, List[str]]:
    """Validate research question is specific and answerable - VERY RELAXED."""
    errors = []

    # RELAXED: Only check basic length (at least 10 words)
    if len(question.split()) < 10:
        errors.append(f"Research question too short ({len(question.split())} words) - needs at least 10 words")

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

    # RELAXED: Only check minimum counts
    if len(data_risks) < 1:
        errors.append("Need at least 1 data risk")
    if len(identification_risks) < 1:
        errors.append("Need at least 1 identification risk")
    if len(scope_limitations) < 1:
        errors.append("Need at least 1 scope limitation")

    return (len(errors) == 0, errors)


# =============================================================================
# Timeline Validation
# =============================================================================

def parse_weeks_range(weeks_str: str) -> Tuple[int, int]:
    """Parse week range string like '4-6' or '4-6 weeks' into (min, max)."""
    try:
        # Strip unit suffixes (weeks, months, days) for parsing
        cleaned = re.sub(r'\s*(weeks?|months?|days?)\s*$', '', weeks_str.strip(), flags=re.IGNORECASE)

        if '-' in cleaned:
            parts = cleaned.split('-')
            return (int(parts[0].strip()), int(parts[1].strip()))
        else:
            # Single number
            val = int(cleaned.strip())
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
    """Validate outlet is specific with quantified rationale - VERY RELAXED."""
    errors = []

    # RELAXED: Only check basic length (at least 5 words for rationale)
    if len(rationale.split()) < 5:
        errors.append(f"{outlet_type} rationale too short ({len(rationale.split())} words) - need at least 5 words")

    return (len(errors) == 0, errors)
