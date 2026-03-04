#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from kgnn_kt.data.io import read_jsonl
from kgnn_kt.kg.build import build_kg_from_problems, save_graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL of problems with fields: problem_id,text,code")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llm", default="mock", help="mock|openai")
    ap.add_argument("--openai_model", default="gpt-4.1-mini")
    args = ap.parse_args()

    problems = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))

    g = build_kg_from_problems(problems, llm_provider=args.llm, openai_model=args.openai_model)
    os.makedirs(args.out_dir, exist_ok=True)
    save_graph(g, os.path.join(args.out_dir, "graph.json"))
    print(f"Saved graph to {os.path.join(args.out_dir, 'graph.json')}")

if __name__ == "__main__":
    main()
