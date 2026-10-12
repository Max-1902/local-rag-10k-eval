#!/usr/bin/env python3
"""Eval harness: measures retrieval hit@k and answer correctness for the
RAG pipeline against a ground-truth question set, and compares against a
no-retrieval (raw model) baseline."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from retrieve import retrieve

RAG_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question using ONLY the "
    "context excerpts provided below. Cite the page number(s) you used. If the context "
    "does not contain the answer, say so explicitly.\n\nContext:\n{context}"
)

NO_RAG_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question about the company's "
    "most recent 10-K filing as best you can from your own knowledge."
)


def format_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[page {h['page']}]\n{h['text']}" for h in hits)


def hit_at_k(hits: list[dict], source_pages: int | list[int], tolerance: int = 1) -> bool:
    """A hit counts if any retrieved chunk's page is within `tolerance` of any valid
    source page (chunks can straddle page boundaries after extraction, and financial
    filings often restate the same figure across multiple sections — MD&A, segment
    notes, percentage-of-revenue tables — so a question can have more than one
    legitimate source page)."""
    if isinstance(source_pages, int):
        source_pages = [source_pages]
    return any(abs(h["page"] - sp) <= tolerance for h in hits for sp in source_pages)


def score_answer(answer: str, item: dict) -> float:
    """Score by substring match. Most questions have one `answer_contains` list
    (fraction of substrings found). Some numeric questions accept alternate
    phrasings (e.g. the model naturally rounds "4,121" to "$4.1 billion") via
    `answer_contains_any`: a list of alternative substring-lists, scored as the
    best-matching alternative."""
    answer_lower = answer.lower()

    def match_fraction(substrings: list[str]) -> float:
        found = sum(1 for s in substrings if s.lower() in answer_lower)
        return found / len(substrings)

    if "answer_contains_any" in item:
        return max(match_fraction(alt) for alt in item["answer_contains_any"])
    return match_fraction(item["answer_contains"])


def run_generation(model, tokenizer, system_prompt: str, question: str, max_tokens: int, sampler) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness")
    parser.add_argument("--model", default="../llama-mlx-quantized")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--questions", required=True, help="python module name holding EVAL_SET, e.g. eval_questions or eval_questions_nvda")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--skip-baseline", action="store_true", help="skip the no-RAG comparison run")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    EVAL_SET = __import__(args.questions).EVAL_SET

    print(f"loading model {args.model} ...")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0, top_p=1.0)

    results = []

    for item in EVAL_SET:
        print(f"\n=== {item['id']} ({item['type']}): {item['question']}")

        hits = retrieve(item["question"], args.collection, k=args.k)
        retrieved_pages = [h["page"] for h in hits]
        hit = hit_at_k(hits, item["source_page"])
        expected = item["source_page"]
        expected_str = f"page {expected}" if isinstance(expected, int) else f"pages {expected}"
        print(f"  retrieved pages: {retrieved_pages} | expected ~{expected_str} | hit@{args.k}={hit}")

        context = format_context(hits)
        rag_answer = run_generation(
            model, tokenizer, RAG_SYSTEM_PROMPT.format(context=context), item["question"], args.max_tokens, sampler
        )
        rag_score = score_answer(rag_answer, item)
        print(f"  RAG answer score: {rag_score:.2f}")

        entry = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "source_page": item["source_page"],
            "retrieved_pages": retrieved_pages,
            "retrieval_hit": hit,
            "rag_answer": rag_answer,
            "rag_score": rag_score,
        }

        if not args.skip_baseline:
            baseline_answer = run_generation(
                model, tokenizer, NO_RAG_SYSTEM_PROMPT, item["question"], args.max_tokens, sampler
            )
            baseline_score = score_answer(baseline_answer, item)
            print(f"  baseline (no RAG) answer score: {baseline_score:.2f}")
            entry["baseline_answer"] = baseline_answer
            entry["baseline_score"] = baseline_score

        results.append(entry)

    # summary
    def avg(key, subset=None):
        rows = [r for r in results if subset is None or r["type"] == subset]
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Retrieval hit@{args.k} overall:   {avg('retrieval_hit'):.2f}")
    print(f"Retrieval hit@{args.k} narrative: {avg('retrieval_hit', 'narrative'):.2f}")
    print(f"Retrieval hit@{args.k} numeric:   {avg('retrieval_hit', 'numeric'):.2f}")
    print(f"RAG answer score overall:   {avg('rag_score'):.2f}")
    print(f"RAG answer score narrative: {avg('rag_score', 'narrative'):.2f}")
    print(f"RAG answer score numeric:   {avg('rag_score', 'numeric'):.2f}")
    if not args.skip_baseline:
        print(f"Baseline answer score overall:   {avg('baseline_score'):.2f}")
        print(f"Baseline answer score narrative: {avg('baseline_score', 'narrative'):.2f}")
        print(f"Baseline answer score numeric:   {avg('baseline_score', 'numeric'):.2f}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
