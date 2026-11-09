import json
from pathlib import Path
import pytest

from healthcare_agent.core import review, resume
from healthcare_agent.evaluation import evaluate


def test_pause_and_resume_human_review():
    case = {"case_id": "SYN-W1", "service": "lumbar MRI", "facts": {"conservative_therapy_weeks": 7, "persistent_symptoms": True}}
    state, graph = review(case)
    assert state["outcome"]["status"] == "pending_human_review"
    assert graph.get_state({"configurable": {"thread_id": "SYN-W1"}}).next == ("human_review",)
    final = resume(graph, "SYN-W1", reviewer_id="reviewer-1", decision="approve", reason="Criteria verified")
    assert final["outcome"]["status"] == "reviewed"
    assert final["outcome"]["reviewer_decision"] == "approve"
    assert final["trace"][-1]["step"] == "human_review"


def test_regression_dataset():
    dataset = Path(__file__).resolve().parents[1] / "src" / "healthcare_agent" / "data" / "eval_cases.json"
    report = evaluate(lambda case: review(case)[0], dataset)
    assert report["passed"] == report["cases"] == 6, json.dumps(report, indent=2)


def test_reviewer_cannot_approve_escalation():
    case = {"case_id": "SYN-W3", "service": "lumbar MRI", "facts": {"red_flags": True}}
    _, graph = review(case)
    with pytest.raises(ValueError, match="Approval is allowed"):
        resume(graph, "SYN-W3", reviewer_id="reviewer-1", decision="approve", reason="Override")


def test_model_proposal_is_corrected_after_guard_critique():
    from healthcare_agent.core import CriticResponse, Proposal, build_graph

    class FakeModel:
        attempts = 0

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, prompt):
            if self.schema is CriticResponse:
                return CriticResponse(issues=[])
            self.attempts += 1
            if self.attempts == 1:
                return Proposal(decision="recommend_approve", rationale="Unsupported", citations=["FAKE@1"])
            return Proposal(decision="request_information", rationale="Need more documentation", citations=["MRI-LSP-01@2026-01"])

    model = FakeModel()
    case = {"case_id": "SYN-W2", "service": "lumbar MRI", "facts": {"persistent_symptoms": True}}
    state, _ = review(case, graph=build_graph(model=model))
    assert model.attempts == 2
    assert state["outcome"]["recommendation"] == "request_information"
    assert [event["step"] for event in state["trace"]].count("critique") == 2
