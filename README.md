# Auditable Prior Authorization Agent

Portfolio engineering prototype for the Deloitte Agentic AI Engineer JD. **Synthetic data and fictional policies only. Not for clinical use, coverage determinations, or PHI.**

## What it demonstrates

- LangGraph stateful workflow: intake → retrieve → proposer → critic → judge → human interrupt/resume.
- Two model roles (proposer and critic) with Pydantic structured output when `--model` is supplied. A deterministic policy guard independently checks proposals and fails closed.
- Bounded self-correction: one revision after critique, then escalation. LangChain `ChatOpenAI` supplies the optional model integration.
- Versioned policy corpus; service-scoped hybrid BM25 and character-trigram retrieval with deterministic reranking and citation validation.
- Explicit human decision with reviewer ID and reason. An audit hash covers the synthetic case and result.
- A six-case regression set checks recommendation, grounding, human gate, and trajectory. CI runs tests and the evaluation gate.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
healthcare-agent examples/mri_case.json
healthcare-agent-eval
pytest -q
```

For the exact environment tested here, install `requirements-lock.txt` first and then `pip install --no-deps -e .`. CI follows that path.

Optional model path:

```bash
pip install -e '.[dev,openai]'
export OPENAI_API_KEY=...
healthcare-agent examples/mri_case.json --model YOUR_MODEL_NAME
```

The CLI shows the **pending** recommendation. A host application must retain the graph/checkpointer and call `resume(graph, case_id, reviewer_id=..., decision=..., reason=...)` after reviewer authentication. No CLI command silently approves a case.

## Review evidence

| Area | Implementation | Verification |
| --- | --- | --- |
| Workflow and pause | `core.py` | `test_workflow.py` in CI |
| Retrieval and citations | `retrieval.py` | `test_decision.py` |
| Fail-closed guard | `decision.py` | red flag and fabricated citation tests |
| Task/trajectory evaluation | `evaluation.py`, packaged `eval_cases.json` | `healthcare-agent-eval` in CI |
| Human oversight | LangGraph interrupt/resume | pause/resume test |

## Limits and production gate

This is **not yet a production healthcare system**. The in-memory checkpointer loses state on restart; a real deployment needs a durable encrypted backend and tested migration/recovery. The policy corpus is illustrative, not validated criteria. No EHR/FHIR or payer integration, PHI processing, authentication, authorization, audit-log storage, rate limiting, telemetry backend, threat model, clinical validation, or HIPAA assessment is included. The audit hash is an integrity comparison aid, not an immutable audit log. Model mode needs live provider tests and latency/cost characterization. The tiny six-case evaluation set does not establish clinical safety.

Before a real deployment: obtain approved policy sources and clinical sign-off; complete security/privacy review and data-flow mapping; implement identity and access controls, durable encrypted checkpoints, immutable audit records, and secret management; test failure/recovery and model/provider outages; evaluate on a representative adjudicated dataset with subgroup and drift analysis; set release thresholds and rollback rules; run supervised pilot with human final authority.

## Design choices

The retrieval layer is intentionally local and deterministic for reproducible offline evaluation. It is a hybrid lexical system, **not embedding/vector retrieval**. The graph keeps model suggestions separate from deterministic guardrails so an unsupported approval cannot pass merely because a model phrases it confidently. Unknown services and red flags escalate. The model never directly invokes an external action.
