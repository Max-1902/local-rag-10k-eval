# Local RAG over SEC 10-K filings

I built a retrieval-augmented generation pipeline that answers questions
about company annual reports and runs entirely on my MacBook: Llama 3.1 8B
Instruct (4-bit, MLX), local embeddings, a local vector store, no API
calls.

Most of my time went into the evaluation. I wrote 65 questions across five
10-K filings, checked every expected lookup answer word for word against
the filing, and traced every lost point back to its cause.

**Report:** [Local-RAG-10K-Eval-Report.pdf](Local-RAG-10K-Eval-Report.pdf)
(5 pages, with charts)

## Results

Pooled over all 65 questions, with 95% bootstrap confidence intervals:

| Metric | Score | 95% CI |
|---|---|---|
| RAG answer score | **0.87** | 0.79 – 0.94 |
| Retrieval hit@5 | **0.92** | 0.86 – 0.98 |
| No-RAG baseline | 0.24 | 0.15 – 0.33 |

Per-company scores are in [rag/PROGRESS.md](rag/PROGRESS.md). With 13
questions per company the intervals are roughly ±0.2, so I don't compare
companies against each other.

What I found:

- Retrieval is necessary. Without it the model scores 0.00 on numeric
  questions for every company, because the filings are newer than its
  training data. That shows why retrieval is needed, not how good it is.
- Retrieval is not the bottleneck. I ran an oracle condition that skips
  retrieval and gives the model only the chunks containing the answer. It
  scores 0.87, the same as normal RAG (difference −0.003, 95% CI ±0.10).
  The lost points come from generation and scoring: the model leaves out a
  detail, refuses when the question names a different table than the one
  it has, or picks the wrong row from tables with repeated labels.
- Redundant context helps. Given only the gold chunk, the model sometimes
  mixed up GAAP and "As adjusted" figures. Normal retrieval avoided this,
  because other retrieved pages restated the correct number.
- Reasoning isn't robust. Computing a margin from two figures went 5/5
  with normal retrieval but 3/5 with the oracle context.
- I found no invented content. I checked every number in all 65 answers
  and screened every sentence against the chunks the model received, then
  read the flagged answers by hand. One perfect score turned out to be a
  false positive.
- Page-level hit@5 was misleading in both directions, so I also checked
  results against the chunk text the model actually received.

The full log, including what broke along the way and how I diagnosed it,
is in [rag/PROGRESS.md](rag/PROGRESS.md).

## How it works

```
PDF ──► PyMuPDF text ──► heading-aware chunks ──► bge-small embeddings ──► Chroma
                                                                             │
question ──► dense search + BM25 ──► reciprocal rank fusion ──► top-5 chunks ┘
                                                                    │
                                     Llama 3.1 8B (4-bit MLX) ◄─────┘
                                                │
                                                ▼
                                     answer with page citations
```

- **Chunking** (`rag/ingest.py`): all-caps headings such as "CASH FLOWS
  STATEMENTS" start a new chunk, so financial tables stay intact and
  labeled. Long narrative sections fall back to a sliding word window.
- **Retrieval** (`rag/retrieve.py`): dense similarity and BM25 keyword
  scores are combined with reciprocal rank fusion. I added BM25 because
  the small embedding model couldn't tell apart line items with similar
  financial wording.
- **Evaluation** (`rag/eval.py`): for each question it records whether a
  top-5 chunk comes from a page that states the answer (±1 page), scores
  the answer by expected substrings, and runs the same question without
  retrieval as a baseline.
- **Oracle** (`rag/eval_oracle.py`): answers each question from the gold
  chunks only, to separate retrieval errors from generation errors.

## Eval set

13 questions per company: 6 narrative, 6 numeric and 1 reasoning. The
questions are in `rag/eval_questions_*.py`, results in
`rag/eval_results_*.json` and oracle results in `rag/eval_oracle_*.json`.

| Company | Filing |
|---|---|
| Microsoft | FY2026 10-K |
| NVIDIA | FY2026 10-K |
| Tesla | FY2025 10-K |
| Lockheed Martin | FY2025 10-K |
| BlackRock | FY2025 10-K |

The filings are public SEC documents and are included in `rag/data/`.

## Running it

You need an Apple Silicon Mac (for MLX) and Python 3.

```bash
python3 -m venv llama-env && source llama-env/bin/activate
pip install -r requirements.txt

# model: needs access to meta-llama/Llama-3.1-8B-Instruct on Hugging Face
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct --mlx-path llama-mlx-quantized -q

cd rag
python3 ingest.py data/MSFT_10K.pdf --collection msft_10k
python3 ingest.py data/NVDA_10K.pdf --collection nvda_10k
python3 ingest.py data/TSLA_10K.pdf --collection tsla_10k
python3 ingest.py data/LMT_10K.pdf  --collection lmt_10k
python3 ingest.py data/BLK_10K.pdf  --collection blk_10k

# evaluate one company
python3 eval.py --model ../llama-mlx-quantized --collection msft_10k --questions eval_questions_MSFT -k 5 --out eval_results_msft.json

# oracle condition: answer from the gold chunks only, no retrieval
python3 eval_oracle.py --collection msft_10k --questions eval_questions_MSFT --compare eval_results_msft.json --out eval_oracle_msft.json

# chat with a filing
python3 rag_chat.py --collection nvda_10k
```

`chat.py` in the repo root is a plain terminal chat with the local model,
without retrieval.

### Reproducibility

- Decoding is greedy (temperature 0) and retrieval is deterministic.
  When I reran evals, they reproduced the stored results exactly.
- I used Python 3.14, mlx 0.30.3 and mlx-lm 0.30.4 (see
  `requirements.txt`). Quantized kernels can differ between MLX versions,
  so other versions may give slightly different answers.
- Model: `meta-llama/Llama-3.1-8B-Instruct`, converted with
  `mlx_lm.convert -q` (4-bit, group size 64). SHA-256 of the resulting
  `model.safetensors`:
  `fe36d962309a5c379bc203996f56a6dde228652d0bb3ed78d4bf60dc9ea7da6c`
- The prompt is `RAG_SYSTEM_PROMPT` in `rag/eval.py`. Every question,
  expected answer and raw model answer is in `rag/eval_questions_*.py`
  and `rag/eval_results_*.json`.

## Limitations

- 13 questions per company is a small sample. One flipped answer moves a
  company's score by about 0.08.
- I wrote both the questions and the system. I adjusted some labels after
  seeing results, following a written rule, and logged every change in
  PROGRESS.md. Every adjustment raised a score, so a held-out question set
  is next.
- Scoring is substring matching. It sometimes marks a correct answer as
  partial because of units ("$3.1 billion" vs. "35,562") or punctuation
  (an Oxford comma). I left these unpatched, so the reported scores are
  conservative.
- A few narrative questions check for keywords that already appear in the
  question, so they can't catch a vague answer (I found one false
  positive).
- hit@5 is measured per page. Whether the answer text appears in the
  retrieved chunks would be a better retrieval metric for this pipeline.

## Tools

I used Claude (Anthropic) as a coding assistant for implementation,
debugging and cross-checking results against the filings, because it let
me audit the evaluation far more thoroughly than I could have by hand in
the same time. The architecture, design decisions and conclusions are my
own.
