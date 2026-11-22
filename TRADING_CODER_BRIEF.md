# Project brief: Fine-tuning a local LLM to write trading strategies

Hand this file to Claude Code at the start of the project. It describes what
to build, the constraints, and what "done" looks like for each phase.

## Goal

Find out how far a small local LLM, fine-tuned on a 16 GB MacBook, can get
at writing **correct** trading-strategy code from a plain-English spec.

This is a portfolio project. The result does not have to be impressive; it
has to be **honestly measured**. The author's previous project (local RAG
over SEC 10-Ks) set the standard: bootstrap confidence intervals, an oracle /
ablation condition, every lost point traced to a cause, and a PROGRESS.md log
of what broke and how it was diagnosed. Keep that standard.

**Headline question:** Does LoRA fine-tuning a 3B coder model on its own
verified outputs close the gap to (or beat) a 7–8B model that hasn't been fine-tuned?

## Hardware constraints (important)

- MacBook Pro, Apple M2, **16 GB unified memory**. Everything runs locally
  with **MLX** (`mlx-lm`). No cloud GPUs, no paid APIs in the pipeline.
- **Use 4-bit models only.** 8-bit 7B+ models leave too little memory.
- Rough budget (4-bit):
  - 1.5B model: ~1 GB weights. Training is easy.
  - 3B model: ~2 GB weights. **The main fine-tuning target.** Comfortable.
  - 7–8B model: ~4.5 GB weights. Fine for inference/baselines. LoRA training
    is a *stretch goal*: batch size 1, `--grad-checkpoint`, ≤8 LoRA layers,
    max sequence length ≤1024. Close other apps; watch memory pressure in Activity Monitor.
- Run one model at a time. Never load two models in the same process.

### Models

Default choices (all from the `mlx-community` org on Hugging Face). Before
starting, check whether newer small coder models (e.g. a newer Qwen coder
generation) exist in 4-bit MLX form and propose swapping if so. Keep the
choice fixed once Phase 2 starts.

| Role | Model |
|---|---|
| Fine-tuning target | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` |
| Small reference | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| Larger reference (no fine-tune) | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` |
| General model (author's previous project) | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |

`mlx-lm` CLI flags change between versions. Check `mlx_lm.lora --help` and
`mlx_lm.generate --help` instead of trusting memory.

## Core design: score correctness, not profit

**Never use backtest profit as the score.** Profit on a handful of price
series is mostly noise and rewards lucky bugs. A strategy is "good" if it does
exactly what the spec says.

### Strategy interface

Every generated solution must define exactly one function:

```python
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    df: DatetimeIndex, columns: open, high, low, close, volume (floats).
    Returns: pd.Series of target positions in {-1, 0, 1}, same index as df.
    Position at bar t may only use data up to and including bar t's close.
    The backtester applies position[t] from bar t+1 onward.
    Bars before the indicator is defined (warm-up) must be 0.
    """
```

- Allowed imports: `numpy`, `pandas`, `math`. Nothing else (no TA-Lib, no
  network, no file I/O).
- State all conventions (warm-up = 0, cross definition, ties, NaN handling)
  explicitly in every prompt. If a convention is ambiguous, the model failing
  is the benchmark's fault, not the model's.

### Sandbox

Run generated code in a subprocess with a timeout (e.g. 10 s), an
import allowlist (check with an AST pass before executing), and no network.
Record the failure type: syntax error, banned import, exception,
timeout, wrong output shape/dtype/values, look-ahead violation, or wrong signals.

### Scoring (per task)

1. **Runs:** executes and returns a Series with the right index and values in {-1,0,1}.
2. **Correct:** output matches a hand-written reference implementation
   exactly on a set of hidden synthetic price series: random walk, strong trend,
   mean-reverting (Ornstein-Uhlenbeck), regime switch, flat stretches, and a
   series with gaps. Generate the series with fixed seeds.
3. **No look-ahead:** pick several random cut points k. Replace all data after
   k with different random data. Positions at bars 0..k must not change. This is
   checked independently of correctness, so a strategy can be "correct" on the
   test series and still be flagged for peeking at future data. Report the
   look-ahead rate per model. This is a headline metric.
4. **Sanity stats (reported, not scored):** Sharpe, max drawdown, turnover, and
   number of trades on one fixed series, only to show strategies behave sensibly.

Primary metric: **pass@1** (greedy decoding) = runs ∧ correct ∧ no look-ahead.
Secondary: pass@5 with sampling (temperature 0.8), run rate, look-ahead rate.
Report 95% bootstrap CIs over tasks for every number.

## Task benchmark

About 100 test tasks, each with: `id`, `family`, `difficulty` (1–3), `prompt`,
`reference.py`, and parameters. Suggested families:

- Moving-average crossovers (SMA/EMA, long-only vs long-short)
- RSI / oscillator thresholds with hysteresis
- Breakouts (Donchian channel, N-day high/low)
- Mean reversion (z-score of price vs rolling mean, entry/exit bands)
- Bollinger Bands
- Volatility filters (only trade when rolling vol < threshold)
- Stop-loss / take-profit / trailing stop (stateful, needs a loop; usually hardest)
- Holding-period rules (exit after N bars)
- Combinations (trend filter + entry signal)

Difficulty 3 should include stateful logic, where most models are expected to fail.

### Splits (prevent leakage)

- **Test-ID (in-distribution):** families seen in training, but parameter
  values and phrasings never used in training data.
- **Test-OOD (out-of-distribution):** 2–3 whole families excluded from training
  entirely (e.g. trailing stops and Bollinger Bands). Tests generalization.
- Write a leakage check that fails the build if any test prompt (normalized)
  or test parameter combination appears in training data.

## Training data

Build a **task generator**: each family is a template with randomized
parameters (window lengths, thresholds, long-only/long-short, etc.) and
several phrasings, plus a programmatic reference implementation. It can
produce thousands of training tasks that never overlap the test sets.

Two training-data sources, compared as an ablation:

- **A. Reference solutions.** Supervised fine-tuning on (prompt → reference
  code). Clean, but the code style is not the model's own.
- **B. Rejection sampling (self-taught).** The base 3B model generates k=8
  samples per training task; keep only samples that pass all checks
  (including look-ahead); dedupe; fine-tune on those. Then repeat
  generate → filter → train for 2–3 rounds. Note that families where the base
  model never passes produce no data. Report this; it is a real limitation.

Format: `mlx-lm` chat JSONL (`{"messages": [...]}`) with `train.jsonl` /
`valid.jsonl`. Keep completions to code only, in one fenced block, and
write an extractor that handles missing or extra fences.

## Phases and done criteria

| # | Phase | Done when |
|---|---|---|
| 0 | Setup | Repo, `pyproject`/`requirements`, `mlx-lm` installed, one model generates text. |
| 1 | Harness | Sandbox, synthetic series, correctness + look-ahead checks. Unit tests prove the checker catches deliberately broken solutions (off-by-one, `shift(-1)`, `center=True` rolling, wrong warm-up) and accepts every reference. |
| 2 | Benchmark | ~100 test tasks with references, Test-ID / Test-OOD splits, leakage check. All references pass their own tests. |
| 3 | Baselines | All 4 models, zero-shot and 3-shot, pass@1 with CIs, failure taxonomy. |
| 4 | Fine-tune A | LoRA on reference solutions (3B). Evaluate on Test-ID and Test-OOD. |
| 5 | Fine-tune B | Rejection-sampling rounds (3B), plotting pass@1 after each round. |
| 6 | Stretch | 7B QLoRA if memory allows; 1.5B to see scaling; prompt-only "self-repair" baseline (feed the error back once) for comparison. |
| 7 | Report | README with results table and charts, a short PDF report, and PROGRESS.md. |

Run a small slice (5–10 tasks) end to end before any full run.

## Suggested repo layout

```
trading-coder/
  README.md
  PROGRESS.md            # running log: decisions, failures, diagnoses
  bench/
    families/            # one module per family: generator + reference
    tasks_test_id.jsonl
    tasks_test_ood.jsonl
  harness/
    sandbox.py           # AST allowlist + subprocess + timeout
    series.py            # seeded synthetic price generators
    checks.py            # run / correctness / look-ahead
    backtest.py          # sanity stats only
  gen/
    make_train.py        # task generator → training tasks
    sample.py            # MLX generation (greedy + sampled)
    filter.py            # rejection sampling → JSONL
  train/
    lora_config.yaml
  eval/
    run_eval.py
    bootstrap.py
    report.py            # tables + charts
  tests/
  results/               # raw generations + scores, one folder per run
```

## Rules for working on this

- Log every run's config (model, adapter, decoding settings, seed, git commit)
  next to its results. Never overwrite results; new run = new folder.
- Keep everything local: no external LLM API calls anywhere in the pipeline.
- Fixed seeds everywhere. Greedy decoding for pass@1.
- Before claiming an improvement, check the CI. With ~100 tasks, differences
  under ~0.1 are probably noise. Say so.
- Read actual failures, not just aggregates. Categorize every failure in a
  sample of at least 30.
- Update PROGRESS.md as you go: what was tried, what broke, how it was diagnosed.
- Ask before long-running jobs (> ~30 min) and say how long you expect them to take.

## Expected interesting findings (hypotheses to test, not claims)

- Fine-tuning mostly fixes *format and convention* errors (warm-up, return
  type, index alignment), less so logic errors.
- Stateful strategies (stops, holding periods) stay hard for everyone.
- Look-ahead bugs are common in base models (e.g. `shift(-1)`, centered
  windows) and fine-tuning on look-ahead-filtered data reduces them.
- Gains on Test-ID are larger than on Test-OOD.
