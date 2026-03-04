from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes.extract_pdf import extract_pdf_node
from agent.nodes.fact_extraction import fact_extraction_node
from agent.nodes.analyze_document import analyze_document_node
from agent.nodes.process_sub_questions import process_sub_questions_node
from agent.nodes.allegation_validation import allegation_validation_node
from agent.nodes.final_synthesis import final_synthesis_node
from agent.nodes.red_team import red_team_node
from agent.nodes.respond import respond_node
from agent.nodes.analyze_followup import analyze_followup_node
from agent.nodes.followup_respond import followup_respond_node


def _has_error(state: AgentState) -> str:
    """Route to respond if there is an error, otherwise continue."""
    if state.get("error"):
        return "respond"
    return "continue"


def build_graph() -> StateGraph:
    """Build the main analysis graph (PDF upload).

    Pipeline:
      extract_pdf
        -> fact_extraction
        -> analyze_document
        -> process_sub_questions
        -> allegation_validation
        -> final_synthesis
        -> red_team
        -> respond
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("extract_pdf", extract_pdf_node)
    workflow.add_node("fact_extraction", fact_extraction_node)
    workflow.add_node("analyze_document", analyze_document_node)
    workflow.add_node("process_sub_questions", process_sub_questions_node)
    workflow.add_node("allegation_validation", allegation_validation_node)
    workflow.add_node("final_synthesis", final_synthesis_node)
    workflow.add_node("red_team", red_team_node)
    workflow.add_node("respond", respond_node)

    # Edges
    workflow.add_edge(START, "extract_pdf")
    workflow.add_conditional_edges(
        "extract_pdf", _has_error,
        {"respond": "respond", "continue": "fact_extraction"},
    )
    workflow.add_conditional_edges(
        "fact_extraction", _has_error,
        {"respond": "respond", "continue": "analyze_document"},
    )
    workflow.add_conditional_edges(
        "analyze_document", _has_error,
        {"respond": "respond", "continue": "process_sub_questions"},
    )
    workflow.add_conditional_edges(
        "process_sub_questions", _has_error,
        {"respond": "respond", "continue": "allegation_validation"},
    )
    workflow.add_conditional_edges(
        "allegation_validation", _has_error,
        {"respond": "respond", "continue": "final_synthesis"},
    )
    workflow.add_conditional_edges(
        "final_synthesis", _has_error,
        {"respond": "respond", "continue": "red_team"},
    )
    workflow.add_edge("red_team", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


def build_followup_graph() -> StateGraph:
    """Build a lighter graph for follow-up questions (no PDF extraction needed).

    Pipeline:
      analyze_followup → process_sub_questions → followup_respond
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("analyze_followup", analyze_followup_node)
    workflow.add_node("process_sub_questions", process_sub_questions_node)
    workflow.add_node("followup_respond", followup_respond_node)
    workflow.add_node("respond", respond_node)  # for error path

    # Edges
    workflow.add_edge(START, "analyze_followup")
    workflow.add_conditional_edges(
        "analyze_followup", _has_error,
        {"respond": "respond", "continue": "process_sub_questions"},
    )
    workflow.add_edge("process_sub_questions", "followup_respond")
    workflow.add_edge("followup_respond", END)
    workflow.add_edge("respond", END)

    return workflow.compile()


graph = build_graph()
followup_graph = build_followup_graph()
