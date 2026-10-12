#!/usr/bin/env python3
"""Terminal chat interface for a local MLX model."""

import argparse
import sys
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel

console = Console()

USER_COLOR = "bold cyan"
ASSISTANT_COLOR = "bold magenta"
DIM = "dim"

HELP_TEXT = """\
[bold]Commands[/bold]
  /help          show this help
  /reset         clear conversation history
  /system <txt>  set a new system prompt (clears history)
  /max <n>       set max tokens per reply (current: {max_tokens})
  /temp <n>      set sampling temperature (current: {temp})
  /exit, /quit   leave the chat
"""


def banner(model_path: str) -> None:
    console.print(
        Panel.fit(
            f"[bold white]local-llama[/bold white] [dim]terminal chat[/dim]\n"
            f"[dim]model:[/dim] {model_path}",
            border_style="magenta",
        )
    )
    console.print("[dim]Type /help for commands.[/dim]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat for a local MLX model")
    parser.add_argument("--model", default="./llama-mlx-quantized", help="path to the MLX model directory")
    parser.add_argument("--max-tokens", type=int, default=2048, help="max tokens generated per reply")
    parser.add_argument("--temp", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="sampling top-p")
    parser.add_argument(
        "--system",
        default="You are a helpful, concise assistant running locally on the user's machine.",
        help="system prompt",
    )
    args = parser.parse_args()

    state = {"max_tokens": args.max_tokens, "temp": args.temp, "top_p": args.top_p}

    with console.status("[bold magenta]Loading model...[/bold magenta]", spinner="dots"):
        model, tokenizer = load(args.model)

    banner(args.model)

    messages = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = console.input(f"[{USER_COLOR}]you ›[/{USER_COLOR}] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")
            cmd = cmd.lower()
            arg = arg.strip()

            if cmd in ("/exit", "/quit"):
                console.print("[dim]bye.[/dim]")
                break
            elif cmd == "/help":
                console.print(HELP_TEXT.format(max_tokens=state["max_tokens"], temp=state["temp"]))
            elif cmd == "/reset":
                messages = [{"role": "system", "content": args.system}]
                console.print("[dim]conversation cleared.[/dim]")
            elif cmd == "/system":
                if arg:
                    args.system = arg
                    messages = [{"role": "system", "content": args.system}]
                    console.print(f"[dim]system prompt set, history cleared.[/dim]")
                else:
                    console.print("[dim]usage: /system <text>[/dim]")
            elif cmd == "/max":
                try:
                    state["max_tokens"] = int(arg)
                    console.print(f"[dim]max_tokens set to {state['max_tokens']}[/dim]")
                except ValueError:
                    console.print("[dim]usage: /max <integer>[/dim]")
            elif cmd == "/temp":
                try:
                    state["temp"] = float(arg)
                    console.print(f"[dim]temp set to {state['temp']}[/dim]")
                except ValueError:
                    console.print("[dim]usage: /temp <float>[/dim]")
            else:
                console.print(f"[dim]unknown command: {cmd} (try /help)[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        sampler = make_sampler(temp=state["temp"], top_p=state["top_p"])

        console.print(f"[{ASSISTANT_COLOR}]assistant ›[/{ASSISTANT_COLOR}] ", end="")

        response_text = ""
        start = time.perf_counter()
        token_count = 0
        try:
            for chunk in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=state["max_tokens"],
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

        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
