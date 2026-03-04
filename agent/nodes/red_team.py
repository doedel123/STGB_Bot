import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState, RedTeamFinding
from prompts.red_team import SYSTEM_PROMPT
from services.gemini_client import llm_with_fallback as llm, extract_text

logger = logging.getLogger(__name__)

_OVERCONFIDENT_TERMS = (
    "zweifelsfrei",
    "eindeutig",
    "ohne jeden zweifel",
    "steht fest",
    "unumstoesslich",
)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _heuristic_findings(state: AgentState) -> list[RedTeamFinding]:
    findings: list[RedTeamFinding] = []
    final_text = (state.get("final_analysis") or "").strip()

    if not final_text:
        return findings

    lines = [ln.strip() for ln in final_text.splitlines() if ln.strip()]

    for line in lines:
        low = line.lower()
        if any(term in low for term in _OVERCONFIDENT_TERMS):
            findings.append(
                {
                    "id": f"RT{len(findings) + 1}",
                    "issue_type": "overconfidence",
                    "severity": "medium",
                    "description": "Moeglich ueberkonfidente Formulierung ohne ausreichende Einschraenkung.",
                    "suggested_fix": "Aussage abschwaechen und mit konkretem Beleg hinterlegen.",
                    "citation": line[:220],
                }
            )
            if len(findings) >= 12:
                return findings

    for line in lines:
        low = line.lower()
        if (
            ("liegt vor" in low or "ist erwiesen" in low or "steht fest" in low)
            and "[" not in line
            and "§" not in line
            and "az" not in low
        ):
            findings.append(
                {
                    "id": f"RT{len(findings) + 1}",
                    "issue_type": "missing_citation",
                    "severity": "high",
                    "description": "Kritische Schlussfolgerung ohne sichtbaren Nachweis.",
                    "suggested_fix": "Seitenzitat aus der Akte oder Rechtsprechungsnachweis nachtragen.",
                    "citation": line[:220],
                }
            )
            if len(findings) >= 12:
                return findings

    validation = state.get("validation_report") or {}
    for allegation_id, entry in validation.items():
        strength = str(entry.get("support_strength") or "none")
        if strength in {"none", "weak"} and allegation_id not in final_text:
            findings.append(
                {
                    "id": f"RT{len(findings) + 1}",
                    "issue_type": "unsupported_claim",
                    "severity": "high" if strength == "none" else "medium",
                    "description": f"{allegation_id} ist {strength} belegt, wird im Endtext aber nicht klar problematisiert.",
                    "suggested_fix": "Im Fehlerbericht explizit als unbelegt markieren und fehlende Tatsachen benennen.",
                    "citation": allegation_id,
                }
            )
            if len(findings) >= 12:
                return findings

    contradictions = state.get("contradictions") or []
    for contradiction in contradictions:
        contradiction_id = contradiction.get("id", "")
        if contradiction_id and contradiction_id not in final_text:
            findings.append(
                {
                    "id": f"RT{len(findings) + 1}",
                    "issue_type": "missed_contradiction",
                    "severity": "medium",
                    "description": f"{contradiction_id} scheint im Endtext nicht explizit adressiert.",
                    "suggested_fix": "Widerspruch im Fehlerbericht als gesonderten Punkt aufnehmen.",
                    "citation": (contradiction.get("description") or "")[:220],
                }
            )
            if len(findings) >= 12:
                return findings

    return findings


def _coerce_finding(item: dict[str, Any], index: int) -> RedTeamFinding:
    issue_type = str(item.get("issue_type") or "unsupported_claim")
    if issue_type not in {
        "missed_contradiction",
        "missing_citation",
        "overconfidence",
        "unsupported_claim",
    }:
        issue_type = "unsupported_claim"

    severity = str(item.get("severity") or "medium")
    if severity not in {"high", "medium", "low"}:
        severity = "medium"

    return {
        "id": str(item.get("id") or f"RT{index}"),
        "issue_type": issue_type,
        "severity": severity,
        "description": str(item.get("description") or "").strip(),
        "suggested_fix": str(item.get("suggested_fix") or "").strip(),
        "citation": str(item.get("citation") or "").strip(),
    }


def _merge_findings(primary: list[RedTeamFinding], secondary: list[RedTeamFinding]) -> list[RedTeamFinding]:
    merged: list[RedTeamFinding] = []
    seen: set[str] = set()

    for collection in (primary, secondary):
        for finding in collection:
            key = (finding.get("issue_type", "") + "|" + finding.get("description", "")).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(finding)

    return merged[:20]


def _append_addendum(final_text: str, findings: list[RedTeamFinding]) -> str:
    if not findings:
        return final_text

    addendum = ["## Addendum: Red-Team Findings"]
    for finding in findings:
        citation = finding.get("citation") or "kein Direktzitat"
        addendum.append(
            f"- [{finding.get('severity', 'medium')}] {finding.get('issue_type', 'issue')}: "
            f"{finding.get('description', '')} | Fix: {finding.get('suggested_fix', '')} | Beleg: {citation}"
        )

    if "Addendum: Red-Team Findings" in final_text:
        return final_text

    return f"{final_text.rstrip()}\n\n" + "\n".join(addendum)


def red_team_node(state: AgentState) -> dict:
    """Adversarially review final synthesis and append actionable addendum."""
    logger.info("red_team: start")

    heuristic_findings = _heuristic_findings(state)
    llm_findings: list[RedTeamFinding] = []

    payload = {
        "final_analysis": state.get("final_analysis", ""),
        "facts": state.get("facts", []),
        "allegations": state.get("allegations", []),
        "validation_report": state.get("validation_report", {}),
        "contradictions": state.get("contradictions", []),
    }

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)[:30000]),
            ]
        )
        parsed = _parse_json_payload(extract_text(response)) or {}
        raw_findings = parsed.get("red_team_findings") if isinstance(parsed.get("red_team_findings"), list) else []
        llm_findings = [
            _coerce_finding(item, i)
            for i, item in enumerate(raw_findings, 1)
            if isinstance(item, dict)
        ]
    except Exception as exc:
        logger.warning("red_team: llm pass failed, using heuristic findings only (%s)", exc)

    findings = _merge_findings(heuristic_findings, llm_findings)
    updated_analysis = _append_addendum(state.get("final_analysis") or "", findings)

    logger.info("red_team: completed findings=%d", len(findings))

    return {
        "red_team_findings": findings,
        "final_analysis": updated_analysis,
    }
