"""Main entry point for the LangGraph application."""

import sys
from pathlib import Path

# Add the graph_app directory to the Python path
graph_app_path = Path(__file__).parent / "graph_app"
sys.path.append(str(graph_app_path))

# Import the create_graph function
from agents.agent import create_graph


def main():
    """Run the LangGraph agent."""
    print("Initializing LangGraph agent...")

    # Create the graph
    graph = create_graph()

    # Define initial state
    initial_state = {
        "message": "Hello, LangGraph!",
        "count": 0
    }

    print("\n" + "="*50)
    print("Running graph with initial state:")
    print(f"  Message: {initial_state['message']}")
    print(f"  Count: {initial_state['count']}")
    print("="*50 + "\n")

    # Run the graph
    result = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("Final state:")
    print(f"  Message: {result['message']}")
    print(f"  Count: {result['count']}")
    print("="*50)


if __name__ == "__main__":
    main()
