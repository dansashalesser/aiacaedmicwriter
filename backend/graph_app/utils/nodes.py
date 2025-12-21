"""Node functions for the graph."""

from utils.state import GraphState


def simple_node(state: GraphState) -> GraphState:
    """
    A simple node that processes the state.
    Add more node functions here as needed.
    """
    print(f"Processing: {state['message']}")

    return {
        "message": f"Processed: {state['message']}",
        "count": state.get("count", 0) + 1
    }
