"""Semantic Scholar API integration for academic paper search."""

import requests
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


def search_papers(
    query: str,
    limit: int = 50,
    fields: Optional[List[str]] = None,
    max_retries: int = 3,
    sort_by_citations: bool = True,
    save_to_json: bool = True
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

    Returns:
        Dictionary containing search results with 'total' count and 'data' list of papers

    Example:
        >>> results = search_papers('machine learning', limit=10)
        >>> print(f"Found {results['total']} papers")
        >>> for paper in results['data']:
        ...     print(paper['title'])
    """
    if fields is None:
        fields = ['title', 'abstract', 'year', 'citationCount', 'authors', 'url', 'publicationDate']

    # Semantic Scholar API has a max limit of 100 per request
    if limit > 100:
        limit = 100

    for attempt in range(max_retries):
        try:
            response = requests.get(
                'https://api.semanticscholar.org/graph/v1/paper/search',
                params={
                    'query': query,
                    'limit': limit,
                    'fields': ','.join(fields)
                }
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
                        x.get('citationCount', 0),
                        x.get('year', 0),
                        x.get('publicationDate', '')
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


if __name__ == "__main__":
    # Example usage
    print("Testing Semantic Scholar search...")
    results = search_papers("quantum computing", limit=5)
    print(format_paper_results(results))
