"""Figures for the RAG report, computed from rag/eval_results_*.json and rag/eval_oracle_*.json.

Needs matplotlib (and reportlab for build_pdf.py), e.g. from the trading-coder venv:
  ../trading-coder/.venv/bin/python rag/report/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RAG = Path(__file__).resolve().parent.parent
FIG = RAG / "report" / "figures"
COMPANIES = ["msft", "nvda", "tsla", "lmt", "blk"]
TYPES = ["narrative", "numeric", "reasoning"]
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#3B6EA5", "#D08C2E", "#9A9A9A", "#222222", "#666666", "#E3E3E3"

# Published 95% bootstrap CIs (README, rag/PROGRESS.md). Point estimates are recomputed below and
# must match; recomputed CIs differ from these by at most 0.01 depending on the bootstrap seed.
PUBLISHED = {"rag": (0.87, 0.79, 0.94), "base": (0.24, 0.15, 0.33), "hit": (0.92, 0.86, 0.98),
             "oracle": (0.87, 0.79, 0.94), "diff": (-0.003, -0.10, 0.10)}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load():
    rows = []
    for c in COMPANIES:
        orc = {o["id"]: o for o in json.loads((RAG / f"eval_oracle_{c}.json").read_text())}
        for r in json.loads((RAG / f"eval_results_{c}.json").read_text()):
            rows.append({**r, "company": c, "oracle_score": orc[r["id"]]["oracle_score"]})
    return rows


def numbers(rows) -> dict:
    mean = lambda v: float(np.mean(v))  # noqa: E731
    out = {"n": len(rows),
           "rag": mean([r["rag_score"] for r in rows]), "base": mean([r["baseline_score"] for r in rows]),
           "hit": mean([float(r["retrieval_hit"]) for r in rows]), "oracle": mean([r["oracle_score"] for r in rows]),
           "better": sum(r["oracle_score"] > r["rag_score"] for r in rows),
           "worse": sum(r["oracle_score"] < r["rag_score"] for r in rows)}
    out["diff"] = out["oracle"] - out["rag"]
    for t in TYPES:
        s = [r for r in rows if r["type"] == t]
        out[t] = {"n": len(s), "hit": mean([r["retrieval_hit"] for r in s]), "rag": mean([r["rag_score"] for r in s]),
                  "base": mean([r["baseline_score"] for r in s]), "oracle": mean([r["oracle_score"] for r in s])}
    out["by_company"] = {c: {"hit": mean([r["retrieval_hit"] for r in rows if r["company"] == c]),
                             "rag": mean([r["rag_score"] for r in rows if r["company"] == c]),
                             "base": mean([r["baseline_score"] for r in rows if r["company"] == c])}
                         for c in COMPANIES}
    for k, (m, _, _) in PUBLISHED.items():
        assert abs(out[k] - m) < 0.006, f"{k}: data {out[k]:.3f} != published {m}"
    return out


def _box(ax, x, y, w, h, title, sub, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=INK, lw=0.8))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.2, color=MUTED, linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, color=INK))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W, H = 0.19, 0.22
    r1, r2, r3 = 0.72, 0.40, 0.06
    _box(ax, 0.0, r1, W, H, "10-K PDF", "text extracted\nwith PyMuPDF")
    _box(ax, 0.26, r1, W, H, "Chunking", "heading-aware;\ntables kept whole")
    _box(ax, 0.52, r1, W, H, "Index", "BGE-small + Chroma,\nand BM25")
    _box(ax, 0.78, r1, 0.21, H, "Hybrid search", "dense + BM25,\nreciprocal rank fusion")
    _box(ax, 0.52, r2, W, H, "Question", "narrative, numeric\nor reasoning")
    _box(ax, 0.78, r2, 0.21, H, "Local LLM", "Llama 3.1 8B, 4-bit,\nMLX; top-5 chunks", fc="#EEF3F9")
    _box(ax, 0.52, r3, 0.47, H, "Evaluation",
         "65 hand-checked questions: retrieval hit@5, answer score,\nno-retrieval baseline and gold-chunk oracle", fc="#FBF1E4")
    for x0, x1 in ((0.19, 0.26), (0.45, 0.52), (0.71, 0.78)):
        _arrow(ax, x0, r1 + H / 2, x1, r1 + H / 2)
    _arrow(ax, 0.71, r2 + H * 0.75, 0.78, r1 + 0.03)
    _arrow(ax, 0.885, r1, 0.885, r2 + H)
    _arrow(ax, 0.885, r2, 0.885, r3 + H)
    ax.text(0.03, r2 + H / 2, "offline: index each filing once\nonline: one query, top-5 chunks,\ngreedy decoding, no API calls",
            fontsize=7.4, color=MUTED, va="center")
    fig.savefig(FIG / "fig1_pipeline.png")
    plt.close(fig)


def fig_main(n):
    rows = [("No retrieval", "model answers from memory", PUBLISHED["base"], GREY),
            ("RAG, top-5 chunks", "normal pipeline", PUBLISHED["rag"], BLUE),
            ("Gold chunks only", "oracle: answer guaranteed in context", PUBLISHED["oracle"], ORANGE)]
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    for i, (lab, sub, (m, lo, hi), col) in enumerate(rows[::-1]):
        ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(m, i, "o", color=col, ms=6, mec="white", mew=0.9)
        ax.text(hi + 0.02, i, f"{m:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=8)
    ax.set_yticks(range(3), [f"{r[0]}\n" for r in rows[::-1]])
    for i, r in enumerate(rows[::-1]):
        ax.text(-0.015, i - 0.22, r[1], ha="right", va="center", fontsize=7, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, 2.5)
    ax.set_xlabel(f"answer score over all {n['n']} questions, 95% bootstrap CI")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig2_main_result.png")
    plt.close(fig)


def fig_by_type(n):
    series = [("Retrieval hit@5", "hit", GREY), ("Answer, no RAG", "base", "#C9C9C9"), ("Answer, RAG", "rag", BLUE),
              ("Answer, gold chunks only", "oracle", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    x = np.arange(len(TYPES))
    w = 0.19
    for j, (lab, key, col) in enumerate(series):
        vals = [n[t][key] for t in TYPES]
        xs = x + (j - 1.5) * w
        ax.bar(xs, vals, w * 0.9, color=col, label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{t.capitalize()} ({n[t]['n']})" for t in TYPES])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=7.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig3_by_type.png")
    plt.close(fig)


def fig_failures():
    causes = [("Answer not retrieved, model correctly declined", 2, GREY),
              ("Page counted as hit, answer chunk not retrieved", 1, GREY),
              ("Figure in a different table than the question named, model refused", 2, BLUE),
              ("Figure was in context, model said it wasn't", 1, BLUE),
              ("Incomplete answer, detail in context left out", 3, BLUE),
              ("Correct answer, scored too strictly by substring match", 3, ORANGE)]
    fig, ax = plt.subplots(figsize=(6.0, 2.1))
    ys = np.arange(len(causes))[::-1]
    for y, (lab, v, col) in zip(ys, causes):
        ax.barh(y, v, color=col, height=0.62)
        ax.text(v + 0.05, y, str(v), va="center", fontsize=8)
    ax.set_yticks(ys, [c[0] for c in causes], fontsize=7.8)
    ax.set_xlim(0, 3.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("lost answers (12 of 60 lookup questions)")
    for lab, col in (("retrieval didn't deliver (3)", GREY), ("in context, model under-used it (6)", BLUE),
                     ("not a real error (3)", ORANGE)):
        ax.bar(0, 0, color=col, label=lab)
    ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=3, fontsize=7.6)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig4_failures.png")
    plt.close(fig)


def fig_oracle_movement():
    better = [("retrieval had truly missed the answer", 2), ("over-literal refusal fixed", 2),
              ("figure overlooked in noisy context", 1), ("sentence left out", 1), ("name on one page only", 1)]
    worse = [("wrong row, repeated table labels", 4), ("needed context outside the chunk", 1),
             ("detail in chunk missed", 2), ("arithmetic slip", 1)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 1.9), gridspec_kw={"wspace": 1.05})
    for ax, items, col, title in ((a, better, ORANGE, "7 answers better with gold chunks"),
                                  (b, worse, BLUE, "8 answers worse with gold chunks")):
        ys = np.arange(len(items))[::-1]
        ax.barh(ys, [v for _, v in items], color=col, height=0.6)
        for y, (_, v) in zip(ys, items):
            ax.text(v + 0.06, y, str(v), va="center", fontsize=7.8)
        ax.set_yticks(ys, [l for l, _ in items], fontsize=7.6)
        ax.set_xlim(0, 4.6)
        ax.set_xticks([0, 2, 4])
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(FIG / "fig5_oracle_movement.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    n = numbers(load())
    fig_pipeline()
    fig_main(n)
    fig_by_type(n)
    fig_failures()
    fig_oracle_movement()
    (RAG / "report" / "numbers.json").write_text(json.dumps(n, indent=1) + "\n")
    print("ok:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
