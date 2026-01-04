"""Semantic similarity checker for proposal deduplication."""

from typing import List, Tuple, Optional, TYPE_CHECKING

# Lazy imports to avoid requiring dependencies at import time
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    import numpy as np

# Global model instance (loaded once, reused across calls)
_model: Optional["SentenceTransformer"] = None


def _get_model() -> "SentenceTransformer":
    """Get or initialize the sentence transformer model (singleton pattern)."""
    from sentence_transformers import SentenceTransformer

    global _model
    if _model is None:
        # Use all-MiniLM-L6-v2: fast, good quality, only 80MB
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def compute_proposal_embedding(proposal) -> "np.ndarray":
    """
    Embed key intellectual components of a proposal.

    Combines:
    - Working title
    - Research question
    - Hypothesis
    - Gap statement (what we don't know)

    These fields capture the intellectual distinctiveness of the proposal.
    """
    import numpy as np

    model = _get_model()

    # Combine key fields that define intellectual distinctiveness
    text_parts = [
        proposal.working_title,
        proposal.research_question,
        proposal.hypothesis,
        proposal.gap_what_we_dont_know
    ]

    # Join with spaces, remove None values
    text = " ".join([part for part in text_parts if part])

    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)

    return embedding


def is_too_similar(
    new_proposal,
    existing_proposals: List,
    threshold: float = 0.75
) -> Tuple[bool, float, int]:
    """
    Check if new proposal is too similar to any existing proposal.

    Args:
        new_proposal: Proposal to check
        existing_proposals: List of already-accepted proposals
        threshold: Similarity threshold (0-1, where 1=identical)

    Returns:
        (is_similar, max_similarity, most_similar_index)
        - is_similar: True if similarity > threshold
        - max_similarity: Highest similarity score found
        - most_similar_index: Index of most similar proposal (-1 if no proposals)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not existing_proposals:
        return (False, 0.0, -1)

    # Compute embedding for new proposal
    new_emb = compute_proposal_embedding(new_proposal)

    # Compute embeddings for existing proposals and calculate similarities
    similarities = []
    for existing in existing_proposals:
        existing_emb = compute_proposal_embedding(existing)
        # Cosine similarity between two vectors
        sim = cosine_similarity([new_emb], [existing_emb])[0][0]
        similarities.append(sim)

    # Find maximum similarity
    max_sim = max(similarities)
    max_idx = similarities.index(max_sim)

    return (max_sim > threshold, max_sim, max_idx)


def generate_differentiation_feedback(
    new_proposal,
    similar_proposal,
    sim_score: float
) -> str:
    """
    Generate feedback for LLM on how to differentiate from similar proposal.

    Args:
        new_proposal: The newly generated proposal
        similar_proposal: The existing similar proposal
        sim_score: Similarity score (0-1)

    Returns:
        Formatted feedback string with specific differentiation instructions
    """
    feedback = f"""
**⚠️ SIMILARITY ALERT**: Your proposal is {sim_score:.1%} similar to an existing proposal.

**Your proposal**: {new_proposal.working_title}
**Your gap**: {new_proposal.gap_what_we_dont_know[:200]}...

**Similar existing proposal**: {similar_proposal.working_title}
**Their gap**: {similar_proposal.gap_what_we_dont_know[:200]}...

**REQUIRED DIFFERENTIATION** - You MUST make substantial changes:

1. **Focus on DIFFERENT assumption/contradiction**:
   - Don't just rephrase the same gap
   - Identify a DIFFERENT untested assumption in the literature
   - Or explore a DIFFERENT unresolved contradiction

2. **Test DIFFERENT mechanism or causal pathway**:
   - Even if studying same domain, examine different theoretical mechanism
   - Propose different mediators, moderators, or boundary conditions

3. **Use DIFFERENT methodological approach or dataset**:
   - Different data source, time period, or geographic scope
   - Different analytical technique or research design
   - Different measurement approach for key constructs

4. **Examine DIFFERENT boundary condition**:
   - Different context where the phenomenon operates
   - Different population, setting, or circumstances
   - Different scope or scale of analysis

**CRITICAL**: Simply rewording is NOT sufficient. The intellectual contribution must be DISTINCT.
Your revised proposal should address a gap that the existing proposal does NOT address.
"""

    return feedback


def compute_pairwise_similarity_matrix(proposals: List) -> "np.ndarray":
    """
    Compute pairwise similarity matrix for a set of proposals.

    Useful for analyzing diversity of a proposal set.

    Args:
        proposals: List of PaperProposal objects

    Returns:
        NxN matrix where entry (i,j) is similarity between proposals i and j
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if not proposals:
        return np.array([])

    # Compute all embeddings
    embeddings = [compute_proposal_embedding(p) for p in proposals]

    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix


def get_average_pairwise_similarity(proposals: List) -> float:
    """
    Compute average pairwise similarity across all proposals.

    Lower scores indicate more diverse proposal set.
    Typical targets:
    - <0.50: Highly diverse
    - 0.50-0.65: Good diversity
    - 0.65-0.75: Moderate similarity
    - >0.75: Too similar, redundant proposals

    Args:
        proposals: List of proposals

    Returns:
        Average similarity score (0-1)
    """
    import numpy as np

    if len(proposals) < 2:
        return 0.0

    sim_matrix = compute_pairwise_similarity_matrix(proposals)

    # Extract upper triangle (excluding diagonal)
    n = len(proposals)
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(sim_matrix[i, j])

    return float(np.mean(upper_triangle))
