#!/usr/bin/env python3
"""RAG-enabled terminal chat: retrieves context from a Chroma collection
before answering, grounding the local MLX model's responses in a document."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.panel import Panel

from retrieve import retrieve

console = Console()

USER_COLOR = "bold cyan"
ASSISTANT_COLOR = "bold magenta"

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question using ONLY the "
    "context excerpts provided below. Each excerpt is labeled with its source page. "
    "Cite the page number(s) you used in your answer. If the context does not contain "
    "the answer, say so explicitly instead of guessing.\n\n"
    "Context:\n{context}"
)


def format_context(hits: list[dict]) -> str:
    parts = []
    for hit in hits:
        parts.append(f"[page {hit['page']}]\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chat over a Chroma collection using a local MLX model")
    parser.add_argument("--model", default="../llama-mlx-quantized", help="path to the MLX model directory")
    parser.add_argument("--collection", required=True, help="chroma collection name to retrieve from")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to retrieve per question")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.3)
    args = parser.parse_args()

    with console.status("[bold magenta]Loading model...[/bold magenta]", spinner="dots"):
        model, tokenizer = load(args.model)

    console.print(
        Panel.fit(
            f"[bold white]RAG chat[/bold white] [dim]grounded on collection '{args.collection}'[/dim]",
            border_style="magenta",
        )
    )
    console.print("[dim]Ask questions about the ingested document. Ctrl+C or /exit to quit.[/dim]\n")

    sampler = make_sampler(temp=args.temp, top_p=1.0)

    while True:
        try:
            question = console.input(f"[{USER_COLOR}]you ›[/{USER_COLOR}] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("/exit", "/quit"):
            console.print("[dim]bye.[/dim]")
            break

        hits = retrieve(question, args.collection, k=args.k)
        context = format_context(hits)

        console.print(f"[dim]retrieved pages: {[h['page'] for h in hits]}[/dim]")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        console.print(f"[{ASSISTANT_COLOR}]assistant ›[/{ASSISTANT_COLOR}] ", end="")

        response_text = ""
        start = time.perf_counter()
        token_count = 0
        try:
            for chunk in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
            ):
                console.print(chunk.text, end="", soft_wrap=True)
                response_text += chunk.text
                token_count = chunk.generation_tokens
        except KeyboardInterrupt:
            console.print("\n[dim]generation interrupted.[/dim]")

        elapsed = max(time.perf_counter() - start, 1e-6)
        console.print()
        console.print(f"[dim]{token_count} tokens · {token_count / elapsed:.1f} tok/s[/dim]\n")


if __name__ == "__main__":
    main()
