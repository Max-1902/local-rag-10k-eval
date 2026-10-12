#!/usr/bin/env python3
"""Oracle eval: answer each question from the gold chunks only (the chunks on
the labeled source page(s) that contain the expected answer), bypassing
retrieval. Comparing this score to the normal RAG score separates retrieval
errors from generation errors."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval import RAG_SYSTEM_PROMPT, format_context, run_generation, score_answer
from retrieve import _get_bm25


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def gold_chunks(item: dict, collection: str, max_chunks: int = 5) -> tuple[list[dict], float]:
    """Greedy cover of the expected substrings using chunks from the source page(s).
    Reasoning questions have no verbatim answer, so they get every source-page chunk."""
    _, ids, docs, metas = _get_bm25(collection)
    pages = item["source_page"] if isinstance(item["source_page"], list) else [item["source_page"]]
    candidates = [
        {"page": m["page"], "text": d}
        for d, m in sorted(zip(docs, metas), key=lambda x: (x[1]["page"], ids[docs.index(x[0])]))
        if m["page"] in pages
    ]

    if item["type"] == "reasoning":
        return candidates[:max_chunks], float("nan")

    alternatives = item.get("answer_contains_any", [item.get("answer_contains")])
    best_chunks, best_cov = [], -1.0
    for subs in alternatives:
        missing = {s.lower() for s in subs}
        chosen = []
        while missing and len(chosen) < max_chunks:
            pick = max(candidates, key=lambda c: sum(s in norm(c["text"]) for s in missing), default=None)
            if pick is None or not any(s in norm(pick["text"]) for s in missing):
                break
            chosen.append(pick)
            missing = {s for s in missing if s not in norm(pick["text"])}
        cov = 1 - len(missing) / len(subs)
        if cov > best_cov:
            best_chunks, best_cov = chosen, cov
    return best_chunks, best_cov


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle (gold-context) eval")
    parser.add_argument("--model", default="../llama-mlx-quantized")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--compare", help="normal eval results JSON to compare against")
    parser.add_argument("--dry-run", action="store_true", help="show gold context coverage without generating")
    parser.add_argument("--out", default="eval_oracle.json")
    args = parser.parse_args()

    eval_set = __import__(args.questions).EVAL_SET
    rag_scores = {}
    if args.compare:
        rag_scores = {r["id"]: r["rag_score"] for r in json.load(open(Path(__file__).parent / args.compare))}

    if not args.dry_run:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        print(f"loading model {args.model} ...")
        model, tokenizer = load(args.model)
        sampler = make_sampler(temp=0.0, top_p=1.0)

    results = []
    for item in eval_set:
        chunks, cov = gold_chunks(item, args.collection)
        line = f"{item['id']} ({item['type']}): {len(chunks)} gold chunk(s) from pages {[c['page'] for c in chunks]}, coverage {cov:.2f}"
        if args.dry_run:
            print(line)
            continue

        answer = run_generation(
            model, tokenizer, RAG_SYSTEM_PROMPT.format(context=format_context(chunks)), item["question"], args.max_tokens, sampler
        )
        score = score_answer(answer, item)
        rag = rag_scores.get(item["id"])
        print(f"{line} | oracle {score:.2f}" + (f" | rag {rag:.2f}" if rag is not None else ""))
        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "oracle_pages": [c["page"] for c in chunks],
                "oracle_coverage": cov,
                "oracle_answer": answer,
                "oracle_score": score,
                "rag_score": rag,
            }
        )

    if args.dry_run:
        return

    def avg(key, subset=None):
        vals = [r[key] for r in results if r[key] is not None and (subset is None or r["type"] == subset)]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 60)
    for subset in (None, "narrative", "numeric", "reasoning"):
        label = subset or "overall"
        msg = f"oracle score {label:9}: {avg('oracle_score', subset):.2f}"
        if rag_scores:
            msg += f"   rag score {label:9}: {avg('rag_score', subset):.2f}"
        print(msg)

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
