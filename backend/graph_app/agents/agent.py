"""Main agent graph construction."""

from langgraph.graph import StateGraph, END
from utils.state import GraphState
from utils.nodes import simple_node


def create_graph():
    """
    Create and return the compiled graph.
    Extend this by adding more nodes and edges.
    """
    # Initialize the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("simple_node", simple_node)

    # Set entry point
    workflow.set_entry_point("simple_node")

    # Add edges
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
