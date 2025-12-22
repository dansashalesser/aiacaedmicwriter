"""Semantic Scholar API integration for academic paper search."""

import requests
import time
import json
import os
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional


def search_papers_sync(
    query: str,
    limit: int = 50,
    fields: Optional[List[str]] = None,
    max_retries: int = 3,
    sort_by_citations: bool = True,
    save_to_json: bool = True,
    min_citation_count: Optional[int] = None,
    publication_types: Optional[List[str]] = None,
    year_range: Optional[str] = None,
    open_access_only: bool = False,
    include_embeddings: bool = False
) -> Dict:
    """
    Search for academic papers using Semantic Scholar API.

    Args:
        query: Search query string (e.g., 'quantum computing')
        limit: Maximum number of papers to return (default: 50, max: 100)
        fields: List of fields to return. Defaults to common useful fields.
        max_retries: Number of times to retry on rate limit errors (default: 3)
        sort_by_citations: Sort results by citation count descending (default: True)
        save_to_json: Save results to JSON file in semantic-scholar folder (default: True)
        min_citation_count: Minimum citation threshold (e.g., 10)
        publication_types: Filter by types like ['Review', 'JournalArticle', 'Conference']
        year_range: Year or year range (e.g., '2023' or '2020-2024')
        open_access_only: Only return papers with public PDFs (default: False)
        include_embeddings: Include specter_v2 embeddings for semantic similarity (default: False)

    Returns:
        Dictionary containing search results with 'total' count and 'data' list of papers

    Example:
        >>> results = search_papers('machine learning', limit=10, min_citation_count=50)
        >>> print(f"Found {results['total']} papers")
        >>> for paper in results['data']:
        ...     print(paper['title'])
    """
    if fields is None:
        fields = ['title', 'abstract', 'year', 'citationCount', 'authors', 'url', 'publicationDate']

    # Add embeddings field if requested
    if include_embeddings and 'embedding.specter_v2' not in fields:
        fields.append('embedding.specter_v2')

    # Semantic Scholar API has a max limit of 100 per request
    if limit > 100:
        limit = 100

    for attempt in range(max_retries):
        try:
            # Build params with optional filters
            params = {
                'query': query,
                'limit': limit,
                'fields': ','.join(fields)
            }

            # Add optional filters
            if min_citation_count is not None:
                params['minCitationCount'] = str(min_citation_count)

            if publication_types:
                params['publicationTypes'] = ','.join(publication_types)

            if year_range:
                params['year'] = year_range

            if open_access_only:
                params['openAccessPdf'] = ''  # Flag parameter

            response = requests.get(
                'https://api.semanticscholar.org/graph/v1/paper/search',
                params=params
            )

            if response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()  # Raise exception for other bad status codes
            results = response.json()

            # Sort by citations, then year, then publication date (all descending)
            if sort_by_citations and results.get('data'):
                results['data'] = sorted(
                    results['data'],
                    key=lambda x: (
                        x.get('citationCount') or 0,
                        x.get('year') or 0,
                        x.get('publicationDate') or ''
                    ),
                    reverse=True
                )

            # Save to JSON if requested
            if save_to_json:
                _save_results_to_json(query, results)

            return results

        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:  # Last attempt
                raise
            time.sleep(2)

    raise Exception("Max retries exceeded")


# Alias for backwards compatibility with existing code (will be removed in Phase 4)
def search_papers(
    query: str,
    limit: int = 50,
    fields: Optional[List[str]] = None,
    max_retries: int = 3,
    sort_by_citations: bool = True,
    save_to_json: bool = True,
    min_citation_count: Optional[int] = None,
    publication_types: Optional[List[str]] = None,
    year_range: Optional[str] = None,
    open_access_only: bool = False,
    include_embeddings: bool = False
) -> Dict:
    """Temporary wrapper for search_papers_sync. Will be removed in Phase 4."""
    return search_papers_sync(
        query, limit, fields, max_retries, sort_by_citations, save_to_json,
        min_citation_count, publication_types, year_range, open_access_only, include_embeddings
    )


async def search_papers_async(
    query: str,
    limit: int = 50,
    fields: Optional[List[str]] = None,
    max_retries: int = 3,
    sort_by_citations: bool = True,
    save_to_json: bool = False,  # Individual searches don't save, only final aggregation
    min_citation_count: Optional[int] = None,
    publication_types: Optional[List[str]] = None,
    year_range: Optional[str] = None,
    open_access_only: bool = False,
    include_embeddings: bool = False
) -> Dict:
    """
    Async version: Search for academic papers using Semantic Scholar API.

    Args:
        query: Search query string (e.g., 'quantum computing')
        limit: Maximum number of papers to return (default: 50, max: 100)
        fields: List of fields to return. Defaults to common useful fields.
        max_retries: Number of times to retry on rate limit errors (default: 3)
        sort_by_citations: Sort results by citation count descending (default: True)
        save_to_json: Save results to JSON file (default: False for individual calls)
        min_citation_count: Minimum citation threshold (e.g., 10)
        publication_types: Filter by types like ['Review', 'JournalArticle', 'Conference']
        year_range: Year or year range (e.g., '2023' or '2020-2024')
        open_access_only: Only return papers with public PDFs (default: False)
        include_embeddings: Include specter_v2 embeddings for semantic similarity (default: False)

    Returns:
        Dictionary containing search results with 'total' count and 'data' list of papers.
        On error, returns: {'total': 0, 'data': [], 'error': str, 'topic': query}
    """
    if fields is None:
        fields = ['title', 'abstract', 'year', 'citationCount', 'authors', 'url', 'publicationDate']

    # Add embeddings field if requested
    if include_embeddings and 'embedding.specter_v2' not in fields:
        fields.append('embedding.specter_v2')

    # Semantic Scholar API has a max limit of 100 per request
    if limit > 100:
        limit = 100

    url = 'https://api.semanticscholar.org/graph/v1/paper/search'

    # Build params with optional filters
    params = {
        'query': query,
        'limit': limit,
        'fields': ','.join(fields)
    }

    # Add optional filters
    if min_citation_count is not None:
        params['minCitationCount'] = str(min_citation_count)

    if publication_types:
        params['publicationTypes'] = ','.join(publication_types)

    if year_range:
        params['year'] = year_range

    if open_access_only:
        params['openAccessPdf'] = ''  # Flag parameter

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        print(f"Rate limited on '{query}'. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status != 200:
                        error_text = await response.text()
                        if attempt == max_retries - 1:
                            return {
                                'total': 0,
                                'data': [],
                                'error': f'HTTP {response.status}: {error_text}',
                                'topic': query
                            }
                        await asyncio.sleep(2)
                        continue

                    results = await response.json()

                    # Sort by citations, then year, then publication date (all descending)
                    if sort_by_citations and results.get('data'):
                        results['data'] = sorted(
                            results['data'],
                            key=lambda x: (
                                x.get('citationCount') or 0,
                                x.get('year') or 0,
                                x.get('publicationDate') or ''
                            ),
                            reverse=True
                        )

                    # Add topic metadata for categorization
                    results['topic'] = query

                    # Note: save_to_json handled by calling node, not individual searches

                    return results

        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                return {
                    'total': 0,
                    'data': [],
                    'error': str(e),
                    'topic': query
                }
            await asyncio.sleep(2)

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'total': 0,
                    'data': [],
                    'error': str(e),
                    'topic': query
                }
            await asyncio.sleep(2)

    # Max retries exceeded
    return {
        'total': 0,
        'data': [],
        'error': 'Max retries exceeded',
        'topic': query
    }


def _save_results_to_json(query: str, results: Dict) -> str:
    """
    Save search results to JSON file with query name and timestamp.

    Args:
        query: The search query string
        results: The results dictionary to save

    Returns:
        Path to the saved JSON file
    """
    # Get the directory where this script is located and create results subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Create sanitized filename from query (remove special chars)
    sanitized_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
    sanitized_query = sanitized_query.replace(' ', '_')

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"{sanitized_query}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filepath}")
    return filepath


def save_categorized_results(
    user_input: str,
    papers_by_topic: Dict[str, List[Dict]],
    all_searches: List[str]
) -> str:
    """
    Save categorized search results to JSON file.

    Args:
        user_input: Original detailed paper idea from user
        papers_by_topic: Papers grouped by search query {query: [papers...]}
        all_searches: List of all searches (original query + topics)

    Returns:
        Path to the saved JSON file

    JSON Structure:
    {
        "user_input": "Original paper idea",
        "original_query": "Original paper idea",
        "topics": ["topic1", "topic2", ...],
        "timestamp": "2025-12-21T10:30:00",
        "total_papers": 45,
        "papers_by_category": {
            "Original paper idea": [{"title": "...", "category": "...", ...}, ...],
            "topic1": [...],
            "topic2": [...]
        }
    }
    """
    # Get the directory where this script is located and create results subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create sanitized filename from user input
    sanitized_input = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in user_input)
    sanitized_input = sanitized_input.replace(' ', '_')[:50]  # Limit length

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"multitopic_{sanitized_input}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    # Calculate total papers
    total_papers = sum(len(papers) for papers in papers_by_topic.values())

    # Separate original query from topics (first item is original query)
    original_query = all_searches[0] if all_searches else user_input
    topics = all_searches[1:] if len(all_searches) > 1 else []

    # Create output structure
    output = {
        "user_input": user_input,
        "original_query": original_query,
        "topics": topics,
        "timestamp": datetime.now().isoformat(),
        "total_papers": total_papers,
        "papers_by_category": papers_by_topic
    }

    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {filepath}")
    return filepath


def _format_paper(output: List[str], index: int, paper: Dict) -> None:
    """Helper function to format a single paper."""
    output.append(f"\n{index}. {paper.get('title', 'No title')}")

    if paper.get('year'):
        output.append(f"   Year: {paper['year']}")

    if paper.get('citationCount') is not None:
        output.append(f"   Citations: {paper['citationCount']}")

    if paper.get('authors'):
        author_names = [author.get('name', '') for author in paper['authors']]
        authors_str = ', '.join(author_names[:3])
        if len(author_names) > 3:
            authors_str += f" (and {len(author_names) - 3} more)"
        output.append(f"   Authors: {authors_str}")

    if paper.get('url'):
        output.append(f"   URL: {paper['url']}")

    if paper.get('abstract'):
        abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
        output.append(f"   Abstract: {abstract}")


def format_categorized_results(papers_by_topic: Dict[str, List[Dict]], user_input: str = None) -> str:
    """
    Format categorized papers for display.

    Args:
        papers_by_topic: Papers grouped by search query/topic
        user_input: Original user input (to identify which is the original query)

    Returns:
        Formatted string with paper information grouped by category
    """
    if not papers_by_topic:
        return "No papers found."

    output = []
    total_papers = sum(len(papers) for papers in papers_by_topic.values())
    num_categories = len(papers_by_topic)

    # Count original query vs topics
    has_original = user_input and user_input in papers_by_topic
    num_topics = num_categories - 1 if has_original else num_categories

    if has_original:
        output.append(f"Found {total_papers} papers: 1 original query + {num_topics} topics\n")
    else:
        output.append(f"Found {total_papers} papers across {num_categories} categories:\n")

    # Show original query first if it exists
    if has_original and user_input in papers_by_topic:
        papers = papers_by_topic[user_input]
        output.append(f"\n{'='*60}")
        output.append(f"ORIGINAL QUERY: {user_input[:80].upper()}")
        output.append(f"{'='*60}")
        output.append(f"Papers: {len(papers)}\n")

        # Show top 5 papers
        for i, paper in enumerate(papers[:5], 1):
            _format_paper(output, i, paper)

        if len(papers) > 5:
            output.append(f"\n   ... and {len(papers) - 5} more papers")

    # Show topics
    for search_query, papers in papers_by_topic.items():
        # Skip original query since we already showed it
        if user_input and search_query == user_input:
            continue

        output.append(f"\n{'='*60}")
        output.append(f"TOPIC: {search_query.upper()}")
        output.append(f"{'='*60}")
        output.append(f"Papers: {len(papers)}\n")

        # Show top 5 papers per topic
        for i, paper in enumerate(papers[:5], 1):
            _format_paper(output, i, paper)

        if len(papers) > 5:
            output.append(f"\n   ... and {len(papers) - 5} more papers")

    return '\n'.join(output)


def format_paper_results(results: Dict) -> str:
    """
    Format paper search results into a readable string.

    Args:
        results: Dictionary returned from search_papers()

    Returns:
        Formatted string with paper information
    """
    if not results.get('data'):
        return "No papers found."

    output = [f"Found {results.get('total', 0)} papers:\n"]

    for i, paper in enumerate(results['data'], 1):
        output.append(f"\n{i}. {paper.get('title', 'No title')}")

        if paper.get('year'):
            output.append(f"   Year: {paper['year']}")

        if paper.get('citationCount') is not None:
            output.append(f"   Citations: {paper['citationCount']}")

        if paper.get('authors'):
            author_names = [author.get('name', '') for author in paper['authors']]
            output.append(f"   Authors: {', '.join(author_names[:3])}")
            if len(author_names) > 3:
                output.append(f" (and {len(author_names) - 3} more)")

        if paper.get('url'):
            output.append(f"   URL: {paper['url']}")

        if paper.get('abstract'):
            abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
            output.append(f"   Abstract: {abstract}")

    return '\n'.join(output)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1 (higher = more similar)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(dot_product / (norm_v1 * norm_v2))


def find_similar_papers(
    reference_paper: Dict,
    candidate_papers: List[Dict],
    top_k: int = 10,
    similarity_threshold: float = 0.0
) -> List[Dict]:
    """
    Find papers most similar to a reference paper using embedding similarity.

    Args:
        reference_paper: Paper with 'embedding' field containing specter_v2 vector
        candidate_papers: List of papers to compare against
        top_k: Number of top similar papers to return (default: 10)
        similarity_threshold: Minimum similarity score to include (default: 0.0)

    Returns:
        List of papers sorted by similarity score (descending), with 'similarity_score' added

    Example:
        >>> results = search_papers_async("machine learning", include_embeddings=True)
        >>> reference = results['data'][0]
        >>> similar = find_similar_papers(reference, results['data'][1:], top_k=5)
        >>> for paper in similar:
        ...     print(f"{paper['title']}: {paper['similarity_score']:.3f}")
    """
    # Extract reference embedding
    ref_embedding = reference_paper.get('embedding', {}).get('vector')
    if not ref_embedding:
        raise ValueError("Reference paper must have 'embedding.vector' field")

    # Calculate similarities
    scored_papers = []
    for paper in candidate_papers:
        paper_embedding = paper.get('embedding', {}).get('vector')
        if not paper_embedding:
            continue

        similarity = cosine_similarity(ref_embedding, paper_embedding)

        if similarity >= similarity_threshold:
            paper_with_score = paper.copy()
            paper_with_score['similarity_score'] = similarity
            scored_papers.append(paper_with_score)

    # Sort by similarity (descending) and return top k
    scored_papers.sort(key=lambda x: x['similarity_score'], reverse=True)
    return scored_papers[:top_k]


def cluster_papers_by_similarity(
    papers: List[Dict],
    similarity_threshold: float = 0.7
) -> List[List[Dict]]:
    """
    Group papers into clusters based on embedding similarity.

    Args:
        papers: List of papers with embeddings
        similarity_threshold: Minimum similarity to be in same cluster (default: 0.7)

    Returns:
        List of paper clusters (each cluster is a list of similar papers)

    Example:
        >>> results = search_papers_async("AI research", limit=50, include_embeddings=True)
        >>> clusters = cluster_papers_by_similarity(results['data'], similarity_threshold=0.75)
        >>> print(f"Found {len(clusters)} research clusters")
        >>> for i, cluster in enumerate(clusters, 1):
        ...     print(f"Cluster {i}: {len(cluster)} papers")
    """
    if not papers:
        return []

    # Filter papers with embeddings (handle None values)
    papers_with_embeddings = [
        p for p in papers
        if p is not None and p.get('embedding', {}).get('vector')
    ]

    if not papers_with_embeddings:
        return []

    clusters = []
    remaining = papers_with_embeddings.copy()

    while remaining:
        # Start new cluster with first remaining paper
        seed = remaining.pop(0)
        cluster = [seed]
        seed_embedding = seed['embedding']['vector']

        # Find all papers similar to seed
        i = 0
        while i < len(remaining):
            candidate = remaining[i]
            candidate_embedding = candidate['embedding']['vector']

            similarity = cosine_similarity(seed_embedding, candidate_embedding)

            if similarity >= similarity_threshold:
                cluster.append(candidate)
                remaining.pop(i)
            else:
                i += 1

        clusters.append(cluster)

    # Sort clusters by size (descending)
    clusters.sort(key=len, reverse=True)
    return clusters


if __name__ == "__main__":
    # Example usage
    print("Testing Semantic Scholar search...")
    results = search_papers_sync("An LLM that helps environmental prediction", limit=5)
    print(format_paper_results(results))
