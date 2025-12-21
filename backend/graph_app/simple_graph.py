"""
Simple LangGraph placeholder with one node.
You can extend this by adding more nodes and edges.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    """State for the graph - add more fields as needed."""
    message: str
    count: int


def simple_node(state: GraphState) -> GraphState:
    """A simple node that processes the state."""
    print(f"Processing: {state['message']}")

    return {
        "message": f"Processed: {state['message']}",
        "count": state.get("count", 0) + 1
    }


def create_graph():
    """Create and return the graph."""
    workflow = StateGraph(GraphState)

    # Add the node
    workflow.add_node("simple_node", simple_node)

    # Set entry point
    workflow.set_entry_point("simple_node")

    # Add edge from node to END
    workflow.add_edge("simple_node", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


if __name__ == "__main__":
    # Create the graph
    graph = create_graph()

    # Run the graph with initial state
    initial_state = {
        "message": "Hello, LangGraph!",
        "count": 0
    }

    print("Running graph...")
    result = graph.invoke(initial_state)

    print("\nFinal state:")
    print(f"Message: {result['message']}")
    print(f"Count: {result['count']}")
