from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SubQuestion(TypedDict):
    question: str
    context_needed: str  # "stgb", "stpo", or "both"
    rag_results: Optional[str]
    search_results: Optional[str]
    synthesis: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: Optional[str]  # concrete user question (None = full analysis)
    pdf_bytes: Optional[bytes]
    pdf_filename: Optional[str]
    pdf_content: Optional[str]
    document_summary: Optional[str]
    sub_questions: list[SubQuestion]
    current_sub_q_index: int
    final_analysis: Optional[str]
    previous_analysis: Optional[str]  # prior analysis for follow-up context
    error: Optional[str]
