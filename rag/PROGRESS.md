# RAG + Eval Project — Progress Log

A local RAG pipeline over SEC filings / earnings materials, running on a
local MLX model (no API calls).

**Goal:** build retrieval, measure whether it actually helps, and document
what breaks and why.

## Contents

**Current state**
1. [Stack](#stack)
2. [Corpus](#corpus)
3. [Eval harness](#eval-harness)
4. [Current results](#current-results-65-questions-pooled)
5. [Next steps](#next-steps)

**Latest analysis (2026-10-11, 13-question sets)**

6. [Question set equalization + reasoning questions](#question-set-equalization--reasoning-question-type-2026-10-11)
7. [Review pass — chunk-level verification](#review-pass--chunk-level-verification-2026-10-11)
8. [Grounding check — all 65 answers](#grounding-check--all-65-answers-2026-10-11)
9. [Oracle condition — gold context only](#oracle-condition--gold-context-only-2026-10-11)

**History (superseded tables, kept for the record)**

10. [Microsoft: building the pipeline](#history-1--microsoft-building-the-pipeline)
11. [NVIDIA](#history-2--nvidia)
12. [Tesla](#history-3--tesla)
13. [Lockheed Martin](#history-4--lockheed-martin)
14. [BlackRock](#history-5--blackrock)
15. [Open problem: numeric/tabular retrieval](#open-problem--numerictabular-retrieval)

---

## Stack

| Component | Choice |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | Heading-aware (financial tables kept intact and labeled with their section heading) + sliding word window fallback for narrative text |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, via sentence-transformers) |
| Vector store | Chroma (local, persistent) |
| Keyword search | BM25 (`rank_bm25`), fused with dense search via reciprocal rank fusion (RRF) |
| Generation | Local MLX Llama 8B-class model (32 layers, hidden size 4096, 4-bit quantized, `llama-mlx-quantized`), reusing the loading/streaming pattern from `chat.py` |

---

## Corpus

Five 10-K filings across four sectors (tech, auto/energy, defense, asset
management):

| Company | Filing | Collection | Chunks |
|---|---|---|---|
| Microsoft | FY2026 10-K | `msft_10k` | 404 |
| NVIDIA | FY2026 10-K | `nvda_10k` | 324 |
| Tesla | FY2025 10-K | `tsla_10k` | 450 |
| Lockheed Martin | FY2025 10-K | `lmt_10k` | 544 |
| BlackRock | FY2025 10-K | `blk_10k` | 594 |

PDFs live in `rag/data/`.

---

## Eval harness

One question file per company (`rag/eval_questions_MSFT.py`,
`eval_questions_nvda.py`, `_tsla`, `_lmt`, `_blk`), each with **13
hand-verified questions**: 6 narrative, 6 numeric, 1 reasoning (65 total).

Every expected answer substring was checked verbatim against the PDF text
before being written, and every source page is a real page that states
the answer.

**Question types**

- **Narrative**: risk factors, business descriptions, legal, HQ, etc.
- **Numeric**: single figures from financial statements. The harder case,
  because answers sit in dense tables.
- **Reasoning**: a margin or percentage the model must compute from two
  retrieved figures (not printed anywhere on the source page).

**Metrics** (`rag/eval.py`, per question)

- **Retrieval hit@5**: did a page that states the answer show up in the
  top 5 retrieved chunks (±1 page tolerance)
- **RAG answer score**: fraction of expected substrings present in the
  model's answer when given retrieved context
- **Baseline answer score**: same, but with no retrieval (model answering
  from its own training knowledge only)

**Run it** (from `rag/`):

```
python3 eval.py --model ../llama-mlx-quantized --collection msft_10k --questions eval_questions_MSFT -k 5 --out eval_results_msft.json
```

---

## Current results (65 questions, pooled)

95% bootstrap confidence intervals over questions:

| Metric | Score | 95% CI |
|---|---|---|
| RAG answer score | **0.87** | 0.79 – 0.94 |
| Retrieval hit@5 | **0.92** | 0.86 – 0.98 |
| No-RAG baseline | 0.24 | 0.15 – 0.33 |

### Per company

13 questions each; intervals are roughly ±0.2, so differences between
companies are **not** meaningful.

| Company | Retrieval hit@5 | RAG answer score | Baseline (no RAG) |
|---|---|---|---|
| Microsoft | 0.85 | 0.74 (0.51 – 0.92) | 0.29 |
| NVIDIA | 1.00 | 0.97 (0.92 – 1.00) | 0.33 |
| Tesla | 0.92 | 0.79 (0.56 – 1.00) | 0.23 |
| Lockheed Martin | 0.92 | 0.87 (0.69 – 1.00) | 0.21 |
| BlackRock | 0.92 | 0.97 (0.92 – 1.00) | 0.13 |

(Bootstrap intervals for NVIDIA and BlackRock are too narrow, because
nearly every score is 1.00. The true uncertainty is similar to the other
companies.)

### Headline findings

Details in the [Review pass](#review-pass--chunk-level-verification-2026-10-11)
and [Grounding check](#grounding-check--all-65-answers-2026-10-11) sections.

- **Why retrieval is needed:** the no-RAG baseline scores 0.00 on numeric
  questions for every company. These filings are newer than the model's
  training data, so this shows the model can't answer from memory. It
  doesn't measure how good retrieval is.

- **Retrieval is not the bottleneck, measured directly.** Giving the model
  only the gold chunks (oracle condition) scores 0.87, the same as normal
  RAG (difference −0.003, 95% CI ±0.10). The remaining losses come from
  generation and scoring.

- **Only 2 of 60 lookup questions** had the answer truly missing from what
  retrieval returned. The rest of the losses happened *after* retrieval
  succeeded: incomplete answers, over-literal refusals, or strict scoring.

- **Redundant context helps.** With only the gold chunk, the model
  sometimes picked the wrong row from tables with repeated labels (GAAP
  vs. "As adjusted"). The normal retrieved context restated the right
  figure and avoided it.

- **Reasoning:** 5/5 correct under normal RAG but 3/5 under the oracle, so
  it isn't robust.

- **All 65 answers** were checked against the chunks the model received:
  no invented figures or unsupported claims. One perfect score is a false
  positive (MSFT n2).

- **Page-level hit@5 was wrong in both directions** about whether the model
  actually had the answer, so this project also checks the retrieved
  chunk text directly.

### Limitations

- n=65 (13 per company).
- The same author wrote the ground truth and the system.
- Labels were adjusted after seeing results, under a written rule, and
  every change raised a score; a held-out set is the real fix.
- Some narrative checks use keywords already present in the question.
- Decoding is greedy, so reruns are identical.

---

## Next steps

- [x] Run the oracle condition for all 5 companies and log results
- [ ] Oracle variant: gold chunks *added to* the normal retrieved context
- [ ] Prompt or chunking fix for duplicate row labels (GAAP vs. As
      adjusted, Sales vs. Cost "Products"): e.g. keep the parent table
      header in every chunk
- [ ] Retrieval ablation on the current 65 questions: dense-only vs.
      BM25-only vs. hybrid (the existing ablation used the old
      12-question Microsoft set)
- [ ] Freeze the question set (git tag), then write and commit a held-out
      batch of questions before running them
- [ ] Add an "answer in retrieved chunks" metric to `eval.py` alongside
      page-level hit@5
- [ ] Decide a scoring policy for paraphrase/hyphenation/unit variants
      (MSFT q1, MSFT n4, BLK n3) and apply it consistently
- [ ] Prompt tweak to test: tell the model the same figure can appear in
      several statements, to reduce over-literal refusals (MSFT q3, TSLA q4)
- [ ] Add more documents: other tickers' 10-Ks, earnings press releases,
      shareholder letters (two-corpus story: narrative-heavy vs.
      numeric/tabular-heavy documents)
- [ ] Grow the eval set beyond 13 questions per company — small N means
      a single flip moves a company's average by ~0.08
- [ ] Try structured table extraction for numeric questions as a third
      retrieval path
- [ ] Still chasing real earnings call transcripts (full Q&A, not just
      press releases) for the narrative/conversational corpus

---

## Question set equalization + reasoning question type (2026-10-11)

Every company's question set was brought to the same size and shape for a
fair cross-company comparison, and one **reasoning** question was added per
company: a question requiring the model to *compute* something (a margin
or percentage) from two retrieved figures, rather than a single-fact
lookup.

All tables in the [History](#history-1--microsoft-building-the-pipeline)
sections reflect the **old, unequal question-set sizes** (12/11/11/11/8)
and are **superseded** by the re-run tables below, run against the
equalized sets (**6 narrative + 6 numeric + 1 reasoning = 13 questions per
company**, verbatim-verified against the PDFs before being written). Old
tables are kept to preserve the history.

All 5 companies have been re-run and analyzed. Several per-company
conclusions below were later corrected by a chunk-level review; see
[Review pass](#review-pass--chunk-level-verification-2026-10-11).

### Re-run — Microsoft

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.85 | 1.00 | 0.67 |
| RAG answer score | 0.74 | 0.89 | 0.56 |
| Baseline answer score | 0.29 | 0.62 | 0.00 |

**Two numeric page misses** (q3 net income per the cash flow statement, q4
dividends paid). Both labeled pages (53 and 54) were re-checked against
the PDF and are correct.

- **q4** is a true retrieval miss: `27,034` is in none of the retrieved
  chunks, and the model correctly declined.
- **q3** is subtler: the cash flow statement wasn't retrieved, but the
  same net income figure (`133,749`) *was* in context via the "Summary
  Results of Operations" table on page 37. The model declined because the
  question named the cash flow statement. That's literally accurate, but
  over-literal.

**q1 (R&D expense increase) is a scoring artifact, not a generation
failure.** The model answered "increased by $3.1 billion or 9%" —
factually correct — but `answer_contains` requires the literal millions
figures (`35,562`, `32,488`), so only `9%` matched → scored 0.33.

Same unit-rescaling false-negative shape as the LMT q4/q6 bug fixed
earlier in this project, just not yet patched for this question. Left
as-is pending a consistent scoring policy; candidate fix is
`answer_contains_any` with a billion-scale alternative.

**r1 (reasoning, operating margin) verified correct.** The model computed
46.8% (FY2026) vs 45.6% (FY2025) from operating income/revenue figures on
page 37, exactly matching the independently-verified ground truth
(155,237/331,839 = 46.78%, 128,528/281,724 = 45.62%). First reasoning
question in the project, held up cleanly.

### Re-run — NVIDIA

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.92 | 1.00 | 0.83 |
| RAG answer score | 0.97 | 0.94 | 1.00 |
| Baseline answer score | 0.33 | 0.72 | 0.00 |

**q2 (R&D expense)** showed as a page miss (page 51 not retrieved), yet
the model answered "$18,497 million (page 41)". **The answer and the
citation are both correct.**

The retrieved page 41 chunk is the MD&A "Operating Expenses" table:
"Research and development $ 18,497 $ 12,914 ...". This is the same
"figure restated in MD&A" labeling artifact as NVDA q1 and BLK q6. Fixed
by setting `source_page` to `[41, 51]`, which makes NVIDIA retrieval
hit@5 1.00 (13/13).

*(Correction: an earlier draft of this entry called q2 a "hallucinated
citation." That came from an off-by-one page check (0-indexed PDF page
41 is printed page 42). The chunk-level review found the figure on the
page the model cited.)*

**r1 (reasoning, net profit margin) verified correct:** 120,067/215,938 =
55.6%, matching the independently-verified ground truth exactly, correctly
cited to page 51 this time (both figures genuinely on that page).

### Re-run — Tesla

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.85 | 1.00 | 0.67 |
| RAG answer score | 0.79 | 0.89 | 0.67 |
| Baseline answer score | 0.23 | 0.50 | 0.00 |

**Two numeric page misses** (q4 stock-based compensation, q6 energy
generation/storage revenue). **In both, the answer was actually in the
retrieved context, and the model still said it wasn't:**

- **q4**: the retrieved page 87 chunk ends with the "Summary Stock-Based
  Compensation Information" table: "Total $ 2,825" for 2025, the same
  figure as the cash flow statement. The model said the context didn't
  contain the *cash flow statement* figure. Over-literal refusal, same
  shape as MSFT q3.
- **q6**: the retrieved page 97 chunk is the segment note: "Energy
  generation and storage segment Revenues $ 12,771". The model answered
  that the exact amount "is not explicitly stated in the given context."
  This is a clear generation-side miss, because the question doesn't
  name a statement. Page 97 is a legitimate alternate source, so
  `source_page` was set to `[53, 97]` (hit@5 becomes True). The rag score
  stays 0.00, because that is what the model actually answered.

**n6 — a chunk-level retrieval miss hidden by a page-level hit, not a
hallucination.**

- Question: what new facility began operations in January 2026?
- The answer is on page 10: *"...in-house lithium refinery in Texas, which
  began operations in January 2026."*
- Page 10 counts as retrieved, but the chunk returned was only its
  "Manufacturing" paragraph, which covers the states and countries part.
  The lithium refinery sentence is in a different page 10 chunk that was
  **not** retrieved.
- The model answered "a residential retrofit solar panel... began initial
  customer deliveries in January 2026." That sentence appears verbatim in
  the retrieved page 7 chunk.
- So the model picked the only "January 2026" fact it had: grounded in
  context, but the wrong kind of thing (a product, not a facility).
  Scored 0.33.

*(Correction: an earlier draft called n6 a fabricated hallucination with
the correct answer in context. That was checked at page level. Chunk
level shows the refinery sentence never reached the model, and its
answer came from a real retrieved passage.)*

**r1 (reasoning, energy revenue as % of total) verified correct:**
12,771/94,827 = 13.5%, matching the independently-verified ground truth
(13.47%) within natural rounding.

### Re-run — Lockheed Martin

First pass (before the n3 ground-truth fix below):

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.85 | 0.83 | 0.83 |
| RAG answer score | 0.82 | 0.78 | 0.83 |
| Baseline answer score | 0.21 | 0.44 | 0.00 |

After fixing n3's ground truth and re-running:

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | **0.92** (was 0.85) | **1.00** (was 0.83) | 0.83 |
| RAG answer score | **0.87** (was 0.82) | **0.89** (was 0.78) | 0.83 |
| Baseline answer score | 0.21 | 0.44 | 0.00 |

**One genuine numeric retrieval miss remains** (q5, stock-based
compensation, page 55 never retrieved) — the model correctly declined
rather than hallucinating.

**n3 (pension funding) — resolved the "unexplained" miss flagged in the
previous LMT run.** Not a retrieval failure or a hallucination:

- The actually-retrieved pages (47, 85) were checked directly against the
  PDF. Both independently and verbatim contain the real funding figures
  ("We made $860 million of cash contributions to our qualified defined
  benefit pension plans in 2025. There are no expected required
  contributions... in 2026").
- The model's answer was fully grounded in retrieved context — it just
  wasn't grounded in the one page (20, abstract risk-factor framing) that
  ground truth had labeled as "the" source.
- Same "financial filings restate the same topic across risk factors /
  MD&A / notes" pattern seen repeatedly (NVDA q1, BLK q6).
- **Fix:** expanded `source_page` to `[20, 47, 85]` and widened
  `answer_contains_any` to accept either the risk-factor vocabulary
  (page 20) or the MD&A dollar figures (pages 47/85), verified verbatim on
  each page before the edit.

**n6 (litigation/environmental exposure) is a real, if minor, generation
imprecision — not a hallucination or labeling bug.**

- Page 25 was correctly retrieved and verbatim contains the required
  sentence: *"We believe the probability is remote that the outcome of
  each of these matters will have a material adverse effect on the
  corporation as a whole..."*
- The model paraphrased this instead of citing it directly ("they cannot
  predict the outcome... with certainty" — actually a different, later
  sentence on the same page), missing the specific "remote"/"material
  adverse effect" language the question was checking for.
- Its "cannot predict the outcome" sentence came from the retrieved page
  22 chunk, which it cited correctly. Scored 0.33 (matched only "party to
  litigation").
- This is a real generation-side partial miss: the key sentence was in
  the retrieved page 25 chunk (confirmed at chunk level) and the model
  left it out. It's an incomplete answer, not an invented one.

**r1 (reasoning, net profit margin) verified correct:** the model computed
5,017/75,048 ≈ 6.68%, matching the independently-verified ground truth
(6.69%) within natural rounding.

### Re-run — BlackRock

First pass (before the n6 and r1 fixes below):

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.92 | 0.83 | 1.00 |
| RAG answer score | 0.87 | 0.89 | 1.00 |
| Baseline answer score | 0.13 | 0.28 | 0.00 |

After fixing n6's ground truth and r1's scoring tolerance and re-running:

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.92 | 0.83 | 1.00 |
| RAG answer score | **0.97** (was 0.87) | **0.94** (was 0.89) | 1.00 |
| Baseline answer score | 0.13 | 0.28 | 0.00 |

Retrieval hit@5 is unchanged (both fixes were answer-scoring/ground-truth
fixes, not retrieval fixes).

**n3 (Preqin acquisition) — the one retrieval miss** (page 12 never
retrieved), but the model still answered correctly from what it did
retrieve.

- At chunk level, the page 40 chunk mentions Preqin and the $3.2 billion
  price, and the page 41/42 chunks mention Preqin.
- The full legal name "Preqin Holding Limited" appears only on page 12, so
  the model said "Preqin," which cost it 1 of 3 substrings (0.67).
- The answer is correct but scored as partial. `source_page` was left at
  12, because no retrieved page contains all expected substrings.

**n6 — found and fixed a ground-truth bug, not a generation failure.**

- The question asks how many *employees* and in how many *countries*.
- Page 40's actual text has two different country counts in one sentence:
  *"approximately 24,900 employees in more than 30 countries... clients in
  more than 100 countries."*
- `answer_contains` had required "100 countries" — the *client*-countries
  figure — paired with the employee count, which is simply the wrong
  number for what the question asks.
- The model correctly answered "more than 30 countries" and was marked
  down for being right. Fixed by correcting `answer_contains` to
  "30 countries".

**r1 (reasoning, net margin) — found and fixed a scoring-tolerance bug,
not a generation failure.**

- The model computed 5,553/24,216 = 23.0% (matches the
  independently-verified ground truth of 22.93% within normal rounding)
  but scored **0.00**.
- `answer_contains_any` listed `"23%"` (no decimal) as an accepted
  alternative, and the model's literal output was `"23.0%"` — `"23%"` is
  not a substring of `"23.0%"`.
- Pure string-matching gotcha, not a math or retrieval error. Fixed by
  adding `"23.0%"` as an explicit accepted alternative.
- Same class of false negative as the LMT q4/q6 unit-rescaling bug.
  **Standing lesson: any time a reasoning/computed-percentage question
  scores 0, read the raw text before trusting the score.**

**No new generation-side failures found for BlackRock** — every low score
traced to either a ground-truth bug (n6), a scoring-tolerance bug (r1), or
a borderline-strict exact-name match (n3), not to the model getting a fact
wrong while holding the right context.

### Cross-company reasoning-question summary

The reasoning question type tests a different failure surface than
factoid lookup: can the model correctly *compute* a derived figure
(margin/percentage) from two retrieved numbers, not just retrieve a
single stated fact.

| Company | Metric | Model answer |
|---|---|---|
| Microsoft | Operating margin | 46.8% / 45.6% |
| NVIDIA | Net profit margin | 55.6% |
| Tesla | Energy revenue % of total | 13.5% |
| Lockheed Martin | Net profit margin | 6.68% |
| BlackRock | Net margin | 23.0% (≈22.93%) |

- Across all 5 companies, **every reasoning question was answered with
  correct arithmetic** under normal RAG, matching independently-verified
  ground truth almost exactly.
- The only reasoning score below 1.00 (BLK r1) traced to an eval *scoring
  bug* (missing decimal-point rounding variant), not to the model
  computing anything incorrectly.
- All five reasoning answers were also checked at chunk level: both input
  figures for each calculation were present in the retrieved chunks, so
  the arithmetic was done on retrieved numbers, not recalled ones.
- *(The oracle condition later showed reasoning is not robust — see
  [Oracle condition](#oracle-condition--gold-context-only-2026-10-11).)*

---

## Review pass — chunk-level verification (2026-10-11)

The retriever returns **chunks**, not pages, but hit@5 and several earlier
checks in this log worked at page level.

- A page can count as "retrieved" while the chunk containing the answer
  was not (TSLA n6).
- A page can also count as "missed" while another retrieved chunk states
  the answer verbatim (NVDA q2, TSLA q4, TSLA q6, MSFT q3).

So every question was re-run through retrieval (results reproduced
exactly) and the expected answer substrings were checked against the
actual chunk text the model received.

**Answer present in retrieved chunks:** 56 of 60 narrative and numeric
questions in full, plus BLK n3 in part (all but the legal name).
Page-level hit@5 on the original labels said 53 of 60.

### Every answer that lost points (12 of 60 non-reasoning questions)

| Cause | Count | Questions |
|---|---|---|
| Answer truly not retrieved; model correctly declined | 2 | MSFT q4, LMT q5 |
| Answer chunk not retrieved despite page "hit"; model answered from another chunk | 1 | TSLA n6 |
| Figure in context from a different table than the question named; model refused | 2 | MSFT q3, TSLA q4 |
| Figure in context; model said it wasn't there | 1 | TSLA q6 |
| Incomplete answer: omitted one required detail that was in context | 3 | MSFT n3, NVDA n6, LMT n6 |
| Scoring artifact: correct answer, strict substring match | 3 | MSFT q1 ("$3.1 billion" vs. raw figures), MSFT n4 ("post-incident reviews" vs. "post incident review"), BLK n3 ("Preqin" vs. full legal name) |

### What this changes about the project's conclusions

- **The earlier claim that "every failure traces to retrieval" doesn't
  hold.** At chunk level, only 3 of 12 point losses come down to retrieval
  not delivering the answer (the first two rows). The rest happened with
  the answer in context or were scoring artifacts.

- **The real generation-side weakness is under-using context, not
  inventing facts.** That means leaving out details, refusing when a
  question names a different table than the one retrieved, and once
  (TSLA q6) not spotting a figure that was there. No answer that lost
  points contained an invented fact; every one traced to retrieved text.
  The two "hallucination" findings in the first draft of this review
  (TSLA n6, NVDA q2) were both wrong, and both errors came from
  page-level checking.

- **Page-level hit@5 is a convenient proxy but measures the wrong unit.**
  "Is the answer text in the retrieved chunks?" is the better retrieval
  metric for this pipeline.

### Ground-truth fixes from this pass

Same rule as before: add an alternate page only if it states the literal
expected answer and the question doesn't name a specific statement.

- NVDA q2 → `[41, 51]`
- TSLA q6 → `[53, 97]`
- MSFT q3 and TSLA q4 were *not* relabeled, because those questions
  explicitly ask for the cash flow statement figure.

Hit@5 values in the "Current results" table reflect these fixes. They
were recomputed with `eval.py`'s own `hit_at_k` over the stored retrieved
pages, which is valid because retrieval reproduced exactly. RAG and
baseline scores are unchanged.

**Left open on purpose (scoring policy, not facts):** MSFT q1, MSFT n4 and
BLK n3 are correct answers scored as partial. Loosening them with
`answer_contains_any` would raise MSFT and BLK scores. It wasn't done
here, so the reported numbers stay conservative.

---

## Grounding check — all 65 answers (2026-10-11)

An outside review pointed out that "no invented facts" only covered the
12 answers that lost points. Substring scoring can't see a fabricated
claim inside an answer that scored 1.00. Two checks closed that gap:

1. **Automated, all 65 answers.** Every number in each answer was checked
   against the text of the 5 chunks the model received, and every
   sentence was scored by how many of its content words appear in those
   chunks.
   - 15 answers were flagged and all 15 read by hand.
   - Every flag was harmless: page citations, years, calculated
     percentages, the "multiply by 100" step in reasoning answers, list
     numbering, or "the context does not contain…" refusals.
   - **No answer contains a figure or claim that isn't in its retrieved
     context.**

2. **Manual, random sample.** 15 of the 53 answers scored 1.00 (fixed
   seed) were read in full against the expected facts.
   - All 15 are correct.
   - One wording slip: MSFT r1 says the margin is "1.2% higher" where
     "1.2 percentage points" is meant.

### One false positive found: MSFT n2

MSFT n2 (LinkedIn Talent/Marketing Solutions) scored 1.00 for a vague
answer that says the details "are not provided." All three expected
substrings ("LinkedIn", "Talent Solutions", "Marketing Solutions") are
already in the question, so any answer that repeats the question passes.

A scan for this weakness found four more narrative questions where half
or more of the expected substrings echo the question: NVDA n3, NVDA n4,
TSLA n2, BLK n1. All four answers were read and are substantively
correct, but those checks can't detect a wrong answer.

Scores were not changed. This is recorded as a limitation of the
question design.

---

## Oracle condition — gold context only (2026-10-11)

The claim that most failures happen *after* retrieval was inferred by
reading cases. `rag/eval_oracle.py` tests it directly.

**Setup**

- Skips retrieval and gives the model only the gold chunks: the chunks on
  the labeled source page(s) that together contain every expected
  substring (greedy cover; for reasoning questions, all source-page
  chunks).
- The prompt, greedy decoding and scoring are identical to `eval.py`.
- A dry run confirmed full answer coverage in the gold context for all 60
  lookup questions.
- Results: `rag/eval_oracle_*.json`.

**Results**

| | Overall | Narrative (30) | Numeric (30) | Reasoning (5) |
|---|---|---|---|---|
| Oracle (gold chunks only) | **0.87** (0.79 – 0.94) | 0.91 | 0.88 | 0.60 |
| Normal RAG | **0.87** (0.79 – 0.94) | 0.91 | 0.81 | 1.00 |

| Company | Oracle | RAG |
|---|---|---|
| Microsoft | 0.79 | 0.74 |
| NVIDIA | 0.97 | 0.97 |
| Tesla | 0.95 | 0.79 |
| Lockheed Martin | 0.90 | 0.87 |
| BlackRock | 0.73 | 0.97 |

**Perfect retrieval did not raise the score.** Paired difference, oracle
minus RAG: −0.003 (95% CI −0.10 to +0.10). With the answer guaranteed to
be in context, the model still scores 0.87. Retrieval is not what limits
this pipeline at n=65; generation and scoring are. This confirms the
earlier inferred claim by measurement, within a ±0.10 margin.

The total hides movement in both directions:

### Better with gold context (7)

MSFT q3, MSFT q4, TSLA q4, TSLA q6, LMT n6, LMT q5, BLK n3:

- the true retrieval misses (MSFT q4, LMT q5)
- the over-literal refusals (MSFT q3, TSLA q4: given the actual cash flow
  statement, the model answers)
- the figure the model overlooked in a noisy context (TSLA q6)
- the omitted sentence (LMT n6)
- the full legal name that sat only on page 12 (BLK n3)

### Worse with gold context (8)

These are new generation errors that the normal RAG context had
prevented.

- **Duplicate row labels.**
  - LMT q3: the earnings table has "Products" under Sales (62,654) and
    again under Operating costs (57,020); the model returned the cost
    line.
  - BLK q4 and q5: page 4 has a GAAP table and an "As adjusted" table
    with identical row labels; the model returned the As-adjusted net
    income ($7,736M) and EPS ($48.09).
  - BLK r1 hit the same confusion and fell into a repetition loop.
- **Missing context outside the chunk.** MSFT n4: the gold chunk never
  names the company, and the model hedged ("Microsoft does not explicitly
  appear in the provided context").
- **Missed or left out a detail that was in the chunk.**
  - BLK n2: the model said the fiscal period "is not specified" although
    the chunk says "For the fiscal year ended December 31, 2025".
  - LMT n2: dropped "U.S. Government" from an otherwise correct
    definition.
  - Neither has an obvious context-composition cause, so these look like
    ordinary generation variance.
- **Arithmetic precision.** MSFT r1 computed 46.76% and 45.59% (correct:
  46.78% and 45.62%). It's close, but the second decimal is wrong. The RAG
  run rounded to one decimal and matched.

In normal RAG, the extra retrieved chunks restated the same figures (e.g.
GAAP net income on pages 43 and 81) and that redundancy steered the model
to the right row. **Redundant context acts as disambiguation**, so a
minimal "perfect" context is not automatically better.

### Corrections this forces on earlier conclusions

- **Reasoning is not robust.** It was 5/5 under normal RAG, but 3/5 under
  the oracle (MSFT r1 arithmetic slip, BLK r1 GAAP/As-adjusted confusion).
  The earlier claim "the model never mis-computed a derived metric" holds
  only for the normal RAG runs.
- **TSLA n6 is confirmed as a chunk-level retrieval miss.** With the
  refinery chunk in context, the oracle answer is fully correct ("an
  in-house lithium refinery, began operations in Texas in January 2026").
  It still scored 0.33 because the scorer expects "California, New York,
  Texas and Nevada" and the model wrote "Texas, and Nevada" (Oxford
  comma). The RAG answer had the same comma, so TSLA n6 also carries a
  punctuation scoring artifact.
- **Persistent misses (below 1.00 in both runs):** MSFT n3 and NVDA n6
  (the model leaves out one detail both times), MSFT n4 and MSFT q1
  (scoring artifacts), TSLA n6 (punctuation artifact).

**Caveat:** this oracle uses the *minimal* gold context (usually one
chunk), which is not the only possible upper bound. Gold chunks added to
the normal retrieved context would separate "retrieval missed it" from
"context composition" more cleanly. That run is a candidate next step.

---

# History

> Everything below uses the **old, unequal question sets** (12/11/11/11/8
> questions) and is **superseded** by the 13-question re-runs above. Kept
> to preserve how the pipeline and the eval methodology evolved.

---

## History 1 — Microsoft: building the pipeline

### Run 1 — dense embeddings only, word-count chunking (baseline RAG)

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.83 | 0.83 | 0.83 |
| RAG answer score | 0.72 | 0.89 | 0.56 |
| Baseline (no RAG) answer score | 0.31 | 0.62 | 0.00 |

**Finding:** RAG roughly doubled answer correctness overall. On numeric
questions the baseline model scored **0.00** across the board — it has no
way to know Microsoft's actual FY2026 figures — while RAG got over half
right.

This is the clearest evidence retrieval is doing real work: the gain is
concentrated exactly where it should be, on facts the model can't
possibly know without a lookup.

**Failures:** q2 (other income, net) and q4 (dividends paid) both scored
0.00 even with retrieval — in both cases the correct chunk wasn't
retrieved at all.

Root cause investigated:

- Fixed 220-word chunking was splitting financial tables apart or merging
  them with unrelated surrounding paragraphs.
- Small financial-statement pages all contain similar generic vocabulary
  ("fiscal year," "millions," "increased/decreased"), so the dense
  embedder couldn't tell them apart.

### Fix attempt 1 — heading-aware chunking

Changed `ingest.py` to detect all-caps section/table headings (e.g.
"OTHER INCOME (EXPENSE), NET", "CASH FLOWS STATEMENTS") and keep each
heading's table as its own chunk, labeled with the heading text, instead
of slicing by raw word count. Chunk count went 357 → 404.

**Verified the fix worked at the chunking level:** inspected the stored
chunk for page 41 directly — the "OTHER INCOME (EXPENSE), NET" table is
now clean and intact as its own chunk.

**But retrieval still failed for q2.** Checked the chunk's rank directly:
it placed **16th** out of candidates.

- Root cause was not chunking — it was **embedding ranking**.
- A generic chunk ("SUMMARY RESULTS OF OPERATIONS", full of dollar figures
  and the phrase "fiscal year") ranked #1 for a query about a specific,
  differently-named line item.
- Small dense embedders don't discriminate well between
  structurally-similar numeric/financial text.

**Takeaway:** fixing chunk boundaries is necessary but not sufficient —
the chunks have to exist *and* rank well. These are two separate failure
modes and need separate fixes.

### Fix attempt 2 — hybrid search (dense + BM25 via RRF)

Added `rank_bm25` keyword scoring alongside dense embedding search in
`retrieve.py`, fused via reciprocal rank fusion (`RRF_K=60`, pool=20
candidates from each method).

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.83 | **1.00** | 0.67 |
| RAG answer score | 0.72 | 0.89 | 0.56 |
| Baseline answer score | 0.31 | 0.62 | 0.00 |

**Finding:** narrative retrieval hit a perfect 1.00 (up from 0.83) —
hybrid search strictly helped there, no downside observed.

**Numeric stayed flat (0.56/0.67) despite changing the architecture** —
but the flat number hides a real swap:

- **q2 (other income) fixed:** 0.00 → 1.00. Hybrid search correctly
  surfaced page 41 (and a near-duplicate table at page 62, footnote Note
  3, which is also legitimately correct content).
- **q3 (net income, cash flow statement) broke:** 1.00 → 0.00. Page 53
  dropped out of the top 5 entirely under hybrid search.

**Debugged q3's regression directly** — checked BM25's raw ranking in
isolation (not just the fused result): page 53 doesn't appear in BM25's
own top 10 either.

- The query says "cash flow **statement**" but the actual filing heading
  is "CASH FLOW**S** STATEMENTS" (plural).
- Words like "statement" / "net income" / "fiscal year" are so common
  across a 10-K (income statement, balance sheet, cash flow statement,
  stockholders' equity statement all share this vocabulary) that neither
  BM25 nor dense embeddings can isolate the one correct table from the
  others using the query as phrased.

**Conclusion:** hybrid search is not a universal fix for numeric retrieval
in dense financial filings. It fails differently from pure dense search,
and the two don't fully overlap in coverage. The net numeric improvement
from this change was zero on this question set, even though the
retrieval mechanism changed and narrative retrieval improved.

---

## History 2 — NVIDIA

**NVIDIA FY2026 10-K** (93 pages, fiscal year ended Jan 25, 2026) —
`rag/data/NVDA_10K.pdf`, collection `nvda_10k`. Ingested with the same
heading-aware chunking + hybrid search pipeline, no changes needed. 324
chunks.

New question set: `rag/eval_questions_nvda.py` (5 narrative + 6 numeric,
same structure as the Microsoft set, verified against actual extracted
text before writing).

`eval.py` was generalized to take `--questions <module>` so either
company's question set can be run against either collection.

### Run — NVIDIA, hybrid search (dense + BM25)

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.82 | 1.00 | 0.67 |
| RAG answer score | **1.00** | 1.00 | 1.00 |
| Baseline (no RAG) answer score | 0.36 | 0.80 | 0.00 |

**Finding:** largest RAG-vs-baseline gap yet (1.00 vs 0.36). Narrative
retrieval is perfect again, consistent with the Microsoft run.

**Caveat — don't take the numeric 1.00 at face value.** q1 (total
revenue) and q2 (R&D expense) both show `hit@5=False` against their
labeled source page (51, the consolidated income statement) — yet both
scored a perfect RAG answer.

- The retrieved chunks (pages 78, 79, 40, 68) weren't *wrong*, they were
  *other valid locations* for the same figures — page 79 is the
  segment/end-market revenue breakdown, page 40 is the
  percentage-of-revenue table, both of which restate or imply the same
  consolidated numbers.
- **This is a ground-truth labeling artifact of financial filings, not a
  retrieval success story:** 10-Ks repeat key figures across MD&A, segment
  notes, and percentage tables, so marking a single page as "the" source
  page under-counts legitimate hits.
- Confirmed this isn't model memorization by directly inspecting the
  retrieved chunk contents — they do contain revenue-adjacent real data,
  just not the exact page I'd pre-labeled.

**Takeaway for the eval methodology:** for financial filings
specifically, `hit@k` scored against a single source page is a stricter
(and sometimes unfair) measure than it would be for a document where each
fact appears exactly once. A more honest version would allow multiple
valid source pages per question.

### Eval methodology fix — multi-page ground truth

Updated `hit_at_k` in `eval.py` to accept a list of valid source pages per
question, not just one (`source_page` can now be `int | list[int]`).

- **NVDA q1:** re-verified directly against the PDF text before changing
  it. Page 79 (revenue-by-end-market note) does restate "Total revenue
  $215,938" verbatim, so it's a legitimate second source, not retrieval
  noise — added `[51, 79]`.
- **NVDA q2:** deliberately did **not** add page 40 as a source for R&D
  expense. Page 40 only has R&D as a percentage-of-revenue (8.6%), not the
  dollar figure the question asks for, so that would have been padding
  the ground truth to manufacture a pass rather than fixing a real
  labeling bug.

**The rule applied:** only add an alternate source page if it
independently contains the literal expected answer substring, verified by
reading the page, not just because retrieval happened to surface it.

### Re-run — NVIDIA, corrected ground truth

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | **0.91** (was 0.82) | 1.00 | **0.83** (was 0.67) |
| RAG answer score | 1.00 | 1.00 | 1.00 |
| Baseline answer score | 0.36 | 0.80 | 0.00 |

Numeric hit@5 corrected from 0.67 → 0.83 — one full question's worth of
the original "miss" was a labeling bug, not a retrieval failure. q2
remains a genuine single miss (R&D expense, page 51 never retrieved).

**Key finding at the time — checked every case for a generation-quality
gap, found none.**

- Every question where retrieval succeeded (`hit@5=True`) had a RAG answer
  score of 1.00, with no exceptions, across both Microsoft and NVIDIA
  runs.
- At that point there was **zero evidence of a generation/extraction
  weakness** at 4-bit quantization — every failure traced to retrieval not
  surfacing the right chunk.
- So switching to a less aggressively quantized model (8-bit, ~9GB vs
  ~4.5GB RAM) would not fix anything broken in the pipeline — all the real
  problems were upstream of generation, in chunking/ranking. No
  model-quantization change planned unless a future document introduces a
  genuine extraction failure.

*(Later superseded by the chunk-level review, which found real
generation-side losses.)*

---

## History 3 — Tesla

**Tesla FY2025 10-K** (139 pages, fiscal year ended Dec 31, 2025) —
`rag/data/TSLA_10K.pdf`, collection `tsla_10k`. 450 chunks. Question set:
`rag/eval_questions_tsla.py` (5 narrative + 6 numeric).

### Run — Tesla, hybrid search (dense + BM25)

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.82 | 1.00 | 0.67 |
| RAG answer score | 0.82 | 1.00 | 0.67 |
| Baseline answer score | 0.27 | 0.60 | 0.00 |

**Pattern holds across a 3rd company:** narrative retrieval perfect again
(1.00), numeric retrieval the weak point again (0.67), RAG answer score
tracks retrieval hit rate almost exactly (0.82/0.82). Every case with a
retrieval hit scored 1.00, every case with a retrieval miss scored 0.00.

**Two numeric misses, both investigated at the time:**

- **q4 (stock-based compensation, cash flow statement, page 56):**
  retrieved pages were all from a ~5-page footnote on the 2025 CEO (Musk)
  Performance Award — a long, detailed note that repeats the exact phrase
  "stock-based compensation expense" far more densely than the single
  summary line in the cash flow statement. A real case of a long,
  keyword-dense footnote out-competing a short, correct table line in
  both BM25 and dense scoring. A new failure mode, distinct from the
  "duplicate table" pattern seen in MSFT/NVDA.
- **q6 (energy generation and storage revenue, page 53):** retrieved the
  MD&A narrative discussion of the same topic (pages 41-42) instead of the
  actual income statement table. Page 41 doesn't contain the dollar figure
  "$12,771"; it only discusses the year-over-year *change* ("$2.69 billion
  increase"). Same prose-vs-table confusion as the very first Microsoft
  finding.

**Methodology note:** before writing these up, checked whether either miss
was a mislabeled ground truth (as happened with NVDA q1) — in both cases,
the "wrong" retrieved pages were checked directly against the PDF text and
confirmed to NOT contain the literal expected figure.

*(Later correction: the chunk-level review found that both figures were in
fact in the retrieved context, on pages not checked here: page 87 has the
stock-comp total $2,825, and page 97 has the segment revenue $12,771. See
[Review pass](#review-pass--chunk-level-verification-2026-10-11).)*

---

## History 4 — Lockheed Martin

**Lockheed Martin FY2025 10-K** (167 pages, Dec 31 2025 fiscal year end,
defense/aerospace — first non-tech company in the corpus) —
`rag/data/LMT_10K.pdf`, collection `lmt_10k`. 544 chunks. Question set:
`rag/eval_questions_lmt.py` (5 narrative + 6 numeric).

### Run — Lockheed Martin, raw result

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.82 | 0.80 | 0.83 |
| RAG answer score | 0.67 | 0.87 | 0.50 |
| Baseline answer score | 0.24 | 0.53 | 0.00 |

This looked like the first break in the pattern held across
MSFT/NVDA/TSLA: q4 and q6 both showed `hit@5=True` **but still scored
0.00** — the first apparent generation-side failures in the whole project.

### Found a scoring bug, not a generation failure

Before logging this, read the actual generated text for both:

- **q4** (cash and cash equivalents, expected "4,121"): model answered
  **"$4.1 billion"**
- **q6** (goodwill, expected "11,314"): model answered **"$11.3 billion"**

Both are correct, naturally-rounded restatements of the right number. The
model chose to state a precise millions figure as a one-decimal billions
figure in prose, which is reasonable and arguably more readable.
`score_answer()` was doing exact substring matching with no concept of
unit rescaling. **This was a false negative in the eval harness itself.**

**Fix:**

- Added an `answer_contains_any` option to the question schema — a list of
  alternative acceptable substring-sets, scored as the best-matching
  alternative (`score_answer` in `eval.py` now takes the full question
  item instead of a bare substring list).
- Applied it only to q4 and q6, only after confirming by reading the raw
  text that the billion-scale phrasing was present and correct — not
  loosened as a blanket policy.
- Verified by re-scoring the already-generated answers directly (no need
  to re-run generation, which is the expensive part) — both flip from 0.0
  to 1.0.

### Run — Lockheed Martin, corrected scores

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.82 | 0.80 | 0.83 |
| RAG answer score | **0.83** (was 0.67) | 0.87 | **0.83** (was 0.50) |
| Baseline answer score | 0.24 | 0.53 | 0.00 |

**With the scoring bug fixed, the pattern held for a 4th company:** every
remaining failure was retrieval-side.

- **q5 (stock-based compensation, page 55 never retrieved):** the one
  true miss left — same "long detailed footnote crowds out short correct
  table line" shape seen with TSLA q4.
- LMT's exhibit-index pages (96/97/98/99, the SEC Item 15 "incorporated by
  reference" list) visibly dominate several numeric queries' retrieved
  results — the exhibit-index noise pattern first seen with MSFT's and
  NVDA's filings shows up a third time.
- **n3 (pension funding)** was this corpus's one real narrative miss —
  retrieved pages (47, 52, 85, 100, 96) all missed page 20. Not
  root-caused at this point (resolved in the equalized re-run).

**Takeaway for the eval harness going forward:** always read the actual
generated text for a 0.00 before accepting the score, especially on
numeric questions — unit rescaling (millions → billions) is a real and
recurring way a correct answer can look wrong to naive substring scoring.

---

## History 5 — BlackRock

**BlackRock FY2025 10-K** (Dec 31, 2025 fiscal year end, asset management
— first financial-services company in the corpus, as opposed to
tech/industrial) — `rag/data/BLK_10K.pdf`, collection `blk_10k`. 594
chunks. Question set: `rag/eval_questions_blk.py` (2 narrative + 6
numeric).

BlackRock reports two genuinely different, similarly-named net income
figures:

- consolidated "Net income" ($5,942M, includes non-controlling interests
  in consolidated investment products)
- "Net income attributable to BlackRock, Inc." ($5,553M, the
  shareholders' share)

q3 and q4 target these on purpose, to test whether retrieval/generation
conflates them.

### Run — BlackRock, raw result

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | 0.88 | 1.00 | 0.83 |
| RAG answer score | 1.00 | 1.00 | 1.00 |
| Baseline answer score | 0.08 | 0.33 | 0.00 |

Only miss: q6 (employee compensation and benefits expense), retrieved
pages `[54, 93, 54, 43, 61]` vs. labeled source page 80 — yet RAG answer
score was still 1.00 (model answered "$8,446 million (page 54)").

**Investigated before logging, per the methodology established with NVDA
q1:** read page 54 and page 80 directly with PyMuPDF.

- Page 80 is the consolidated income statement.
- Page 54 is a *different*, full standalone expense table inside the MD&A
  "Investment Advisory, Administration Fees and Securities Lending
  Revenue" discussion section.
- Both independently state "$8,446" / "$6,546" for 2025/2024 employee
  compensation and benefits — the literal dollar figure in both places.
- Same "financial filings repeat key figures" labeling artifact as NVDA q1.
  Added page 54 to `source_page` (now `[54, 80]`).

### Run — BlackRock, corrected ground truth

| Metric | Overall | Narrative | Numeric |
|---|---|---|---|
| Retrieval hit@5 | **1.00** (was 0.88) | 1.00 | **1.00** (was 0.83) |
| RAG answer score | 1.00 | 1.00 | 1.00 |
| Baseline answer score | 0.08 | 0.33 | 0.00 |

**First corpus with a perfect 1.00 across every metric.** Caveat:

- BlackRock's MD&A happens to duplicate every queried figure in a second,
  independently-retrievable location (the full expense/revenue breakdown
  table appears both in the financial statements and restated in MD&A).
- This is likely a property of BlackRock's filing structure rather than
  evidence the retrieval pipeline improved. Treat as a ceiling effect, not
  an architecture win — no code changed between the LMT and BLK runs.

**q3 vs. q4 net income ambiguity — checked explicitly, no confusion
found.**

- Both questions' top-5 retrieved sets include pages for *both* net income
  figures (q3 retrieved `[4, 81, 43, 82, 44]`, q4 retrieved
  `[4, 43, 81, 118, 47]` — pages 4 and 81 appear in both), so the model had
  both numbers in context for both questions.
- Generation still correctly picked $5,942M for q3 ("consolidated...
  before attribution to non-controlling interests") and $5,553M for q4
  ("attributable to BlackRock, Inc.") with no mixing.
- The phrasing distinction in the question was enough for the model to
  disambiguate, even with both figures in the same context window. The
  generation-side disambiguation risk this question pair was designed to
  probe did not materialize.

**Pattern across 5 companies at the time:** narrative retrieval perfect in
4/5 runs (LMT's 0.80 the lone exception); RAG answer score had zero
generation-side failures once scoring bugs and ground-truth labeling
artifacts were corrected.

*(Later superseded by the chunk-level review and oracle condition.)*

---

## Open problem — numeric/tabular retrieval

Numeric/tabular retrieval in financial filings is a known hard case for
both lexical and semantic search, because:

1. The real distinguishing information (which exact table) is in the
   heading/label, not spread through the numbers
2. Financial-statement vocabulary is highly repetitive across a filing
3. Query phrasing often doesn't match the filing's exact heading wording

The standard production fix is **not** more retrieval tuning — it's
**structured extraction at ingestion time**: parse financial tables into
explicit key-value pairs, e.g.

```
{statement: "cash flow", line_item: "net income", fiscal_year: 2026, value: 133749}
```

and query that structure directly for numeric questions, falling back to
text retrieval only for narrative questions.
