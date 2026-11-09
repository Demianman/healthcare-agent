import json
from pathlib import Path

import pytest

from healthcare_agent.decision import critique, policy_recommendation, validate_case
from healthcare_agent.retrieval import HybridRetriever, load_corpus


CORPUS = Path(__file__).resolve().parents[1] / "src" / "healthcare_agent" / "data" / "policies.json"


def evidence(service):
    return HybridRetriever(load_corpus(CORPUS)).search(service, service)


def test_retrieval_is_service_scoped_and_versioned():
    rows = evidence("lumbar MRI")
    assert [r["citation"] for r in rows] == ["MRI-LSP-01@2026-01"]
    assert evidence("unknown service") == []


def test_red_flags_override_approval():
    case = {"case_id": "SYN-1", "service": "lumbar MRI", "facts": {"red_flags": True, "conservative_therapy_weeks": 9, "persistent_symptoms": True}}
    assert policy_recommendation(case, evidence(case["service"]))["decision"] == "escalate"


def test_unsupported_model_proposal_fails_closed():
    case = {"case_id": "SYN-2", "service": "lumbar MRI", "facts": {"persistent_symptoms": True}}
    proposal = {"decision": "recommend_approve", "rationale": "Looks fine", "citations": ["FAKE@1"]}
    assert "unsupported citation" in critique(case, evidence(case["service"]), proposal)
    assert "recommendation conflicts with deterministic policy guard" in critique(case, evidence(case["service"]), proposal)


@pytest.mark.parametrize("bad", [
    {"case_id": "REAL-123", "service": "lumbar MRI", "facts": {}},
    {"case_id": "SYN-1", "service": "lumbar MRI", "facts": {"patient_name": "Jane"}},
    {"case_id": "SYN-1", "service": "lumbar MRI", "facts": {"conservative_therapy_weeks": True}},
])
def test_rejects_invalid_or_phi_like_inputs(bad):
    with pytest.raises(ValueError):
        validate_case(bad)
