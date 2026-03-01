from langchain_core.messages import AIMessage

from agent.state import AgentState


def respond_node(state: AgentState) -> dict:
    """Format the final analysis (or error) as a chat message."""
    if state.get("error"):
        content = f"Fehler bei der Analyse: {state['error']}"
    else:
        content = state.get("final_analysis", "Keine Analyse verfuegbar.")

    # Safety: ensure content is a plain string
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        content = "".join(parts)

    return {"messages": [AIMessage(content=content)]}
