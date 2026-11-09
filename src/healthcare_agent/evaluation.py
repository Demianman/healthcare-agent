"""Case and trajectory regression evaluation, independent of model/provider."""
from __future__ import annotations

import json
from pathlib import Path


def evaluate(run_case, dataset: Path) -> dict:
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    results = []
    for item in cases:
        state = run_case(item["case"])
        outcome = state["outcome"]
        steps = [event["step"] for event in state["trace"]]
        valid_citations = {row["citation"] for row in state["evidence"]}
        checks = {
            "expected_recommendation": outcome["recommendation"] == item["expected"],
            "human_review": outcome["human_review_required"] and outcome["status"] == "pending_human_review",
            "grounding": set(outcome["citations"]).issubset(valid_citations),
            "trajectory": steps[:2] == ["intake", "retrieve"] and steps[-2:] == ["critique", "judge"],
        }
        results.append({"case_id": item["case"]["case_id"], "checks": checks, "passed": all(checks.values())})
    return {"cases": len(results), "passed": sum(r["passed"] for r in results), "results": results}
