import json
from pathlib import Path

from .core import review
from .evaluation import evaluate


def main():
    dataset = Path(__file__).resolve().parent / "data" / "eval_cases.json"
    report = evaluate(lambda case: review(case)[0], dataset)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["cases"] else 1)


if __name__ == "__main__":
    main()
