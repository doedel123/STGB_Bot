from __future__ import annotations

from typing import Annotated, Optional, TypedDict, NotRequired

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Citation(TypedDict, total=False):
    id: str
    source_type: str  # document | commentary | case_law | analysis
    source: str
    uri: str
    page: str
    quote: str
    note: str


class Fact(TypedDict, total=False):
    id: str
    text: str
    page: str
    page_range: str
    quote: str


class Allegation(TypedDict, total=False):
    id: str
    text: str
    type: str  # legal_conclusion | factual_claim | inference
    page: str
    page_range: str
    quote: str


class Contradiction(TypedDict, total=False):
    id: str
    description: str
    involved_fact_ids: list[str]
    involved_allegation_ids: list[str]
    evidence_quotes: list[str]


class ValidationEntry(TypedDict, total=False):
    supporting_fact_ids: list[str]
    support_strength: str  # strong | medium | weak | none
    notes: str
    circular_reasoning: bool


class RedTeamFinding(TypedDict, total=False):
    id: str
    issue_type: str  # missed_contradiction | missing_citation | overconfidence | unsupported_claim
    severity: str  # high | medium | low
    description: str
    suggested_fix: str
    citation: str


class SubQuestion(TypedDict):
    question: str
    context_needed: str  # "stgb", "stpo", or "both"
    rag_results: Optional[str]
    search_results: Optional[str]
    synthesis: Optional[str]
    issues_to_check: NotRequired[list[str]]


class AgentState(TypedDict):
    provider: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: Optional[str]  # concrete user question (None = full analysis)
    pdf_bytes: Optional[bytes]
    pdf_filename: Optional[str]
    pdf_content: Optional[str]
    raw_text: Optional[str]
    document_structure: NotRequired[dict]
    document_summary: Optional[str]
    sub_questions: list[SubQuestion]
    current_sub_q_index: int
    facts: list[Fact]
    allegations: list[Allegation]
    contradictions: list[Contradiction]
    validation_report: dict[str, ValidationEntry]
    red_team_findings: list[RedTeamFinding]
    citations: list[Citation]
    issues_to_check: list[str]
    final_analysis: Optional[str]
    previous_analysis: Optional[str]  # prior analysis for follow-up context
    error: Optional[str]
