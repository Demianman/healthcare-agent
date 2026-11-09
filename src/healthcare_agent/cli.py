import argparse
import json
import os
from pathlib import Path

from .core import build_graph, review


def main():
    parser = argparse.ArgumentParser(description="Review a synthetic prior-authorization case")
    parser.add_argument("case", type=Path)
    parser.add_argument("--model", help="OpenAI model name; requires OPENAI_API_KEY and [openai] extra")
    args = parser.parse_args()
    model = None
    if args.model:
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is required with --model")
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=args.model, temperature=0, timeout=30, max_retries=2)
    state, _ = review(json.loads(args.case.read_text()), graph=build_graph(model=model))
    print(json.dumps({"outcome": state["outcome"], "trace": state["trace"], "evidence": state["evidence"]}, indent=2))


if __name__ == "__main__":
    main()
