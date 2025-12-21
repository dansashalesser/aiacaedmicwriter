"""State definition for the graph."""

from typing import TypedDict


class GraphState(TypedDict):
    """
    State for the graph.
    Add more fields as needed for your application.
    """
    message: str
    count: int
