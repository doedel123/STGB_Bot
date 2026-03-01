from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes.extract_pdf import extract_pdf_node
from agent.nodes.analyze_document import analyze_document_node
from agent.nodes.retrieve_rag import retrieve_rag_node
from agent.nodes.search_case_law import search_case_law_node
from agent.nodes.synthesize_step import synthesize_step_node
from agent.nodes.final_synthesis import final_synthesis_node
from agent.nodes.respond import respond_node


def _has_error(state: AgentState) -> str:
    """Route to respond if there is an error, otherwise continue."""
    if state.get("error"):
        return "respond"
    return "continue"


def _check_progress(state: AgentState) -> str:
    """Route back to retrieve_rag if more sub-questions remain, else to final_synthesis."""
    if state["current_sub_q_index"] < len(state["sub_questions"]):
        return "retrieve_rag"
    return "final_synthesis"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("extract_pdf", extract_pdf_node)
    workflow.add_node("analyze_document", analyze_document_node)
    workflow.add_node("retrieve_rag", retrieve_rag_node)
    workflow.add_node("search_case_law", search_case_law_node)
    workflow.add_node("synthesize_step", synthesize_step_node)
    workflow.add_node("final_synthesis", final_synthesis_node)
    workflow.add_node("respond", respond_node)

    # Edges
    workflow.add_edge(START, "extract_pdf")
    workflow.add_conditional_edges(
        "extract_pdf", _has_error, {"respond": "respond", "continue": "analyze_document"}
    )
    workflow.add_conditional_edges(
        "analyze_document", _has_error, {"respond": "respond", "continue": "retrieve_rag"}
    )
    workflow.add_edge("retrieve_rag", "search_case_law")
    workflow.add_edge("search_case_law", "synthesize_step")
    workflow.add_conditional_edges(
        "synthesize_step", _check_progress, {"retrieve_rag": "retrieve_rag", "final_synthesis": "final_synthesis"}
    )
    workflow.add_edge("final_synthesis", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


graph = build_graph()
