"""Fail-closed policy checks; model suggestions cannot override these controls."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


ALLOWED_FACTS = {
    "conservative_therapy_weeks", "persistent_symptoms", "progressive_neurological_deficit",
    "red_flags", "functional_limitation", "clinician_plan",
}


def validate_case(case: dict) -> None:
    if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not re.fullmatch(r"SYN-[A-Za-z0-9_-]{1,64}", case["case_id"]):
        raise ValueError("Only synthetic case IDs of the form SYN-<token> are accepted")
    if not isinstance(case.get("service"), str) or not case["service"].strip():
        raise ValueError("service must be a nonempty string")
    facts = case.get("facts")
    if not isinstance(facts, dict) or set(facts) - ALLOWED_FACTS:
        raise ValueError("facts contain unknown or invalid fields")
    if any(not isinstance(facts[k], bool) for k in facts if k != "conservative_therapy_weeks"):
        raise ValueError("Boolean facts must be true or false")
    weeks = facts.get("conservative_therapy_weeks", 0)
    if isinstance(weeks, bool) or not isinstance(weeks, int) or not 0 <= weeks <= 520:
        raise ValueError("conservative_therapy_weeks must be an integer from 0 to 520")
    if len(json.dumps(case)) > 5000:
        raise ValueError("Case exceeds size limit")


def policy_recommendation(case: dict, evidence: list[dict]) -> dict:
    facts = case["facts"]
    citations = {row["id"]: row["citation"] for row in evidence}
    service = case["service"].lower()
    if facts.get("red_flags"):
        return {"decision": "escalate", "rationale": "Red flags require clinical review.", "citations": []}
    if service == "lumbar mri" and "MRI-LSP-01" in citations:
        met = facts.get("progressive_neurological_deficit") is True or (facts.get("conservative_therapy_weeks", 0) >= 6 and facts.get("persistent_symptoms") is True)
        return {"decision": "recommend_approve" if met else "request_information",
                "rationale": "MRI-LSP-01 criteria evaluated against documented facts.", "citations": [citations["MRI-LSP-01"]]}
    if service == "knee physical therapy" and "PT-KNEE-02" in citations:
        met = facts.get("functional_limitation") is True and facts.get("clinician_plan") is True
        return {"decision": "recommend_approve" if met else "request_information",
                "rationale": "PT-KNEE-02 criteria evaluated against documented facts.", "citations": [citations["PT-KNEE-02"]]}
    return {"decision": "escalate", "rationale": "No applicable policy found.", "citations": []}


def critique(case: dict, evidence: list[dict], proposal: dict) -> list[str]:
    issues = []
    allowed_decisions = {"recommend_approve", "request_information", "escalate"}
    if proposal.get("decision") not in allowed_decisions:
        issues.append("invalid decision")
    citations = proposal.get("citations")
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        issues.append("invalid citations")
        citations = []
    known = {row["citation"] for row in evidence}
    if not set(citations).issubset(known):
        issues.append("unsupported citation")
    if proposal.get("decision") == "recommend_approve" and not citations:
        issues.append("approval recommendation requires evidence")
    expected = policy_recommendation(case, evidence)
    if proposal.get("decision") != expected["decision"]:
        issues.append("recommendation conflicts with deterministic policy guard")
    if proposal.get("decision") == "recommend_approve" and not set(expected["citations"]).issubset(citations):
        issues.append("applicable policy citation missing")
    return issues


def audit_hash(case: dict, outcome: dict) -> str:
    payload = json.dumps({"case": case, "outcome": outcome}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
