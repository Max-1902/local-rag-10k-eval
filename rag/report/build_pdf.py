"""Build Local-RAG-10K-Eval-Report.pdf (plain technical-report layout, same design as my trading-coder report).

Usage (needs matplotlib + reportlab):
  python rag/report/make_figures.py && python rag/report/build_pdf.py
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

RAG = Path(__file__).resolve().parent.parent
ROOT = RAG.parent
FIG = RAG / "report" / "figures"
OUT = ROOT / "Local-RAG-10K-Eval-Report.pdf"
AUTHOR = "Maximilian Wojciechowski"
DATE = "October 2026 (revised November 2026)"
REPO = "github.com/Max-1902/local-rag-10k-eval"
N = json.loads((RAG / "report" / "numbers.json").read_text())

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6)
TITLE = ParagraphStyle("title", fontName="Times-Bold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Times-Italic", fontSize=11.5, leading=14, alignment=TA_CENTER, spaceAfter=10)
AUTH = ParagraphStyle("auth", fontName="Times-Roman", fontSize=11, leading=14, alignment=TA_CENTER)
ABS_H = ParagraphStyle("absh", fontName="Times-Bold", fontSize=10.5, alignment=TA_CENTER, spaceBefore=14, spaceAfter=4)
ABS = ParagraphStyle("abs", parent=BODY, fontSize=10, leading=13, leftIndent=0.45 * inch, rightIndent=0.45 * inch)
H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=12.5, leading=15, spaceBefore=12, spaceAfter=5)
H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=10.8, leading=13.5, spaceBefore=8, spaceAfter=3)
CAP = ParagraphStyle("cap", fontName="Times-Roman", fontSize=9, leading=11.2, alignment=TA_JUSTIFY, spaceBefore=3,
                     spaceAfter=10)
TCAP = ParagraphStyle("tcap", parent=CAP, spaceBefore=6, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.8, leading=10.6, alignment=TA_LEFT)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Times-Bold")
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=9.2, leftIndent=12, spaceAfter=8)

story: list = []
counters = {"fig": 0, "tab": 0}


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def h1(text):
    story.append(Paragraph(text, H1))


def h2(text):
    story.append(Paragraph(text, H2))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))
    story.append(Spacer(1, 3))


def figure(name, caption, width=6.2 * inch):
    counters["fig"] += 1
    iw, ih = ImageReader(str(FIG / name)).getSize()
    story.append(KeepTogether([Spacer(1, 4), Image(str(FIG / name), width=width, height=width * ih / iw),
                               Paragraph(f"<b>Figure {counters['fig']}.</b> {caption}", CAP)]))


def table(rows, caption, widths, bold_rows=()):
    counters["tab"] += 1
    data = [[Paragraph(str(c), CELLB if i == 0 or i in bold_rows else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(KeepTogether([Paragraph(f"<b>Table {counters['tab']}.</b> {caption}", TCAP), t, Spacer(1, 8)]))


def code(text):
    story.append(KeepTogether([Preformatted(text.strip("\n"), CODE)]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(letter[0] / 2, 0.55 * inch, str(doc.page))
    canvas.restoreState()


f2 = lambda x: f"{x:.2f}"  # noqa: E731
T = {t: N[t] for t in ("narrative", "numeric", "reasoning")}

# ===================================================================== content
story.append(Spacer(1, 0.25 * inch))
p("Local RAG over SEC 10-K Filings: Measuring Where a Fully Local Pipeline Breaks", TITLE)
p("A student project on retrieval, an oracle test and failure analysis with a 4-bit Llama 3.1 8B", SUB)
p(f"{AUTHOR}<br/>{DATE} · {REPO}", AUTH)

p("Abstract", ABS_H)
p("I built a retrieval-augmented generation (RAG) pipeline that answers questions about the annual reports (10-K "
  "filings) of Microsoft, NVIDIA, Tesla, Lockheed Martin and BlackRock, and runs entirely on my MacBook: a 4-bit "
  "Llama 3.1 8B through MLX, local embeddings, and no API calls. I tested it on 65 questions whose answers I checked "
  "by hand against the PDFs. With retrieval the answer score is 0.87 [0.79, 0.94]; without retrieval it is 0.24, "
  "and 0.00 on every numeric question, because the filings are newer than the model's training data. To find out "
  "whether retrieval limits the pipeline, I ran an oracle condition that gives the model only the chunks that "
  "contain the answer. It also scores 0.87 (paired difference −0.003 [−0.10, +0.10]): at this sample size "
  "retrieval is necessary but not the bottleneck. Reading every lost answer showed that in half of them the right "
  "text was already in the model's context, that page-level retrieval metrics are wrong in both directions, and "
  "that more context can beat perfect context.", ABS)

h1("1 Introduction")
p("Annual reports are long, dense and full of tables, and they are published after a language model's training "
  "cut-off, so the model has to read them rather than remember them. That makes them a good test for RAG. My goal "
  "was not just to build a working pipeline but to measure where it breaks, using a local model on my own laptop. "
  "The question I wanted to answer is simple: when the system gets an answer wrong, is it because retrieval failed "
  "or because the model failed to use what it was given?")
p("What I built and found:")
bullets([
    "A fully local pipeline: PDF extraction, heading-aware chunking that keeps financial tables whole, hybrid search "
    "(embeddings plus BM25), and a 4-bit Llama 3.1 8B.",
    "A hand-verified evaluation set of 65 questions (13 per company) covering narrative, numeric and reasoning "
    "questions, with a no-retrieval baseline and a gold-chunk oracle.",
    "Retrieval is necessary (0.24 → 0.87) but not the bottleneck: perfect retrieval scores the same.",
    "A failure analysis of every lost answer, which also uncovered three bugs in my own evaluation.",
])

h1("2 Pipeline")
figure("fig1_pipeline.png", "The pipeline. Each filing is indexed once; at question time hybrid search returns the "
       "top five chunks, which go into the prompt of the local model.")
p("Text is extracted with PyMuPDF. Chunking follows the document's headings, and each financial table stays in one "
  "chunk labelled with its heading, so a number never gets separated from the row and column that give it meaning. "
  "Chunks are embedded with BGE-small and stored in Chroma, and in parallel indexed for BM25 keyword search. The two "
  "rankings are merged by reciprocal rank fusion, and the top five chunks go into the prompt of Llama 3.1 8B "
  "Instruct (4-bit, MLX), decoded greedily so reruns are identical.")
p("<b>How it evolved.</b> I started with dense search only, then added heading-aware chunking, then hybrid search. "
  "On an earlier, smaller Microsoft question set, hybrid search took narrative retrieval from 0.83 to 1.00, but the "
  "numeric answer score stayed at 0.56: it fixed one question and broke another. Changing the retrieval algorithm "
  "did not move that number, which is what made me look past retrieval in the first place.")

h1("3 Evaluation")
table([["Type", "Questions", "What it asks"],
       ["Narrative", "30 (6 per company)", "risk factors, business description, legal matters"],
       ["Numeric", "30 (6 per company)", "one figure from a financial statement; the hard case, because answers sit in "
        "dense tables"],
       ["Reasoning", "5 (1 per company)", "compute a margin or share from two retrieved figures; the result is not "
        "printed anywhere"]],
      "The 65 evaluation questions, 13 per company.", [0.9, 1.3, 4.1])
p("I checked every expected answer for the lookup questions word for word against the PDF and computed the reasoning "
  "answers from it; later label fixes are logged. I report three measures:")
bullets([
    "<b>Retrieval hit@5:</b> one of the top five chunks comes from a page that states the answer (±1 page).",
    "<b>Answer score:</b> the share of expected facts that appear in the answer.",
    "<b>No-retrieval baseline:</b> the same model and question without retrieved context.",
])
p("All intervals are 95% bootstrap confidence intervals over questions.")

h1("4 Results")
figure("fig2_main_result.png", "Answer score over all 65 questions: 0.24 without retrieval, 0.87 with it, and still "
       "0.87 with perfect retrieval. Dot = score, line = 95% bootstrap CI.", width=5.6 * inch)
table([["Measure", "Score", "95% CI"],
       ["RAG answer score", "0.87", "0.79 – 0.94"],
       ["Retrieval hit@5", "0.92", "0.86 – 0.98"],
       ["No-retrieval baseline", "0.24", "0.15 – 0.33"],
       ["Gold chunks only (oracle)", "0.87", "0.79 – 0.94"]],
      "Pooled results over all 65 questions.", [2.4, 0.9, 1.3])
p("The no-retrieval row matters for interpretation. These filings are newer than the model's training data, so it "
  f"cannot answer from memory: it scores {f2(T['numeric']['base'])} on every numeric question and "
  f"{f2(T['narrative']['base'])} on narrative ones, where general knowledge about a company sometimes overlaps with "
  "the filing. That shows retrieval is necessary, but not how good it is. Section 5 answers the second question.")
figure("fig3_by_type.png", "Retrieval and answer scores by question type. Without retrieval the model answers no "
       "numeric or reasoning question.")
bc = N["by_company"]
names = {"msft": "Microsoft", "nvda": "NVIDIA", "tsla": "Tesla", "lmt": "Lockheed Martin", "blk": "BlackRock"}
cis = {"msft": "0.51 – 0.92", "nvda": "0.92 – 1.00", "tsla": "0.56 – 1.00", "lmt": "0.69 – 1.00",
       "blk": "0.92 – 1.00"}
table([["Company", "Retrieval hit@5", "RAG answer score (95% CI)", "No retrieval"]] +
      [[names[c], f2(bc[c]["hit"]), f"{f2(bc[c]['rag'])} ({cis[c]})", f2(bc[c]["base"])] for c in bc],
      "Per company, 13 questions each. Intervals are about ±0.2, so I do not compare companies; the intervals for "
      "NVIDIA and BlackRock are too narrow because nearly every score is 1.00.", [1.4, 1.1, 2.0, 1.0])
p("Scores are conservative: a few correct answers still lose points to strict substring matching (Section 6).")

h1("5 Oracle test: perfect retrieval")
p("At first, “the failures happen after retrieval” was something I inferred by reading cases. To measure it, I "
  "skipped retrieval and gave the model only the gold chunks, the ones that contain the answer, with the same "
  "prompt, the same greedy decoding and the same scoring.")
table([["", "All (65)", "Narrative (30)", "Numeric (30)", "Reasoning (5)"],
       ["RAG, top-5 chunks", "0.87", f2(T["narrative"]["rag"]), f2(T["numeric"]["rag"]), f2(T["reasoning"]["rag"])],
       ["Gold chunks only", "0.87", f2(T["narrative"]["oracle"]), f2(T["numeric"]["oracle"]), f2(T["reasoning"]["oracle"])]],
      "Answer score, normal RAG versus gold chunks only.", [1.6, 0.9, 1.1, 1.1, 1.1])
p("The paired difference, gold minus RAG, is −0.003 (95% CI −0.10 to +0.10). With the answer guaranteed to be "
  "in context the model still scores 0.87, so at this sample size retrieval is not what limits the pipeline; "
  f"generation and scoring are. The flat total hides movement in both directions: {N['better']} answers got better "
  f"with gold context and {N['worse']} got worse.")
figure("fig5_oracle_movement.png", "Why answers changed when the model received only the gold chunks instead of the "
       "normal top five.")
p("<b>More context beat perfect context.</b> BlackRock reports a GAAP table and an “As adjusted” table with "
  "identical row labels. Given only the gold chunk, the model returned the as-adjusted net income, $7,736M instead of "
  "$5,553M. In normal RAG, other retrieved pages restated the GAAP figure, and that redundancy steered it to the "
  "right row. This oracle uses the minimal gold context, usually one chunk. Adding the gold chunks on top of normal "
  "retrieval would separate “retrieval missed it” from “context composition” more cleanly; that is the "
  "next run.")

h1("6 Failure analysis")
p("Under normal RAG, 12 of the 60 lookup answers lost points. I read each one and traced it to a cause. Only three "
  "were retrieval problems. In six, the right text was in the model's context and it still left out a detail, "
  "refused too literally, or missed a figure. For example, Tesla's segment note with energy revenue of $12,771M was "
  "in context, and the model said the amount wasn't stated.")
figure("fig4_failures.png", "Causes of the 12 lost lookup answers under normal RAG. Half had the right text in "
       "context.")
h2("6.1 Page-level hit@5 is wrong in both directions")
table([["Counting method", "Lookup questions"],
       ["Right page retrieved (original labels)", "53 / 60"],
       ["Answer text actually in what the model received", "56 / 60"]],
      "The 60 lookup questions, counted two ways.", [3.6, 1.4])
p("Four answers were in context even though their labelled page was not retrieved, because 10-Ks repeat figures in "
  "several places. One page counted as a hit, but the chunk with the answer never arrived. From then on I checked "
  "chunks, not pages.")
h2("6.2 Grounding check")
p("For all 65 answers I checked every number and screened every sentence against the five chunks the model received, "
  "and read the flagged ones by hand. I found no invented figures or claims. One perfect score was a false positive: "
  "a vague answer passed because the expected keywords were already in the question.")

h1("7 Reasoning questions")
table([["Company", "Metric", "With RAG", "Gold only", "Ground truth"],
       ["MSFT", "Operating margin, FY26 / FY25", "46.8% / 45.6%", "46.76% / 45.59%", "46.78% / 45.62%"],
       ["NVDA", "Net margin", "55.6%", "correct", "55.60%"],
       ["TSLA", "Energy share of revenue", "13.5%", "correct", "13.47%"],
       ["LMT", "Net margin", "6.68%", "correct", "6.69%"],
       ["BLK", "Net margin", "23.0%", "GAAP / adjusted mix-up", "22.93%"]],
      "Computing a margin or share from two figures in the filing: 5 of 5 with retrieval, 3 of 5 with gold context only.",
      [0.7, 1.9, 1.1, 1.4, 1.1])
p("The reasoning results are not robust. The same model goes from five correct to three correct when the context "
  "changes, and the BlackRock failure is the same repeated-label problem as in Section 5.")

h1("8 How I checked my own evaluation")
p("Reading every lost answer, not just the totals, turned up problems in my evaluation as well as in the pipeline:")
table([["Problem", "How I found it", "What I changed"],
       ["Millions vs. billions false negative", "reading lost answers", "fixed the expected value"],
       ["“23%” vs. “23.0%” substring miss", "reading lost answers", "fixed the scoring rule"],
       ["Wrong country count in the ground truth", "re-checking against the PDF", "corrected the label"],
       ["Two “hallucination” calls that were wrong", "checked whole pages, once the wrong one",
        "check the chunks the model actually saw"],
       ["Page-level hit@5 misleading", "comparing pages with chunk text", "report chunk-level evidence"],
       ["A perfect score that was a false positive", "auditing the 1.00s too", "noted; keywords came from the question"]],
      "Problems in my own evaluation and how I fixed them. All label changes are logged in rag/PROGRESS.md.",
      [2.4, 1.9, 2.1])

h1("9 Takeaways")
bullets([
    "<b>Retrieval is necessary, not the bottleneck.</b> 0.24 without it, 0.87 with it, and perfect retrieval adds "
    "nothing on top. What is left is generation and scoring.",
    "<b>More context can beat perfect context.</b> Redundant chunks restate figures and keep the model off the wrong "
    "row; a minimal “perfect” context made 8 answers worse and 7 better.",
    "<b>Check chunks, not pages, and audit the wins too.</b> Page-level hit@5 was wrong in both directions, and "
    "reading the perfect scores turned up a false positive.",
])

h1("10 Limitations and next steps")
bullets([
    "n = 65 (13 per company), and I wrote both the questions and the system.",
    "Labels were adjusted after seeing results, under a written rule, and every change raised a score; a held-out "
    "set is planned.",
    "Substring scoring misses some correct paraphrases, and a few narrative checks reuse keywords from the question.",
    "Decoding is greedy, so reruns are identical; there is no sampling variance to report.",
])
p("Next, I would add the gold chunks on top of normal retrieval, keep the parent table header in every chunk so "
  "repeated row labels stay unambiguous, then re-run dense versus BM25 versus hybrid on the current 65 questions, "
  "freeze the set, and write a held-out batch before running it.")

h1("Tools and acknowledgements")
p("I used Claude Code as an AI coding assistant while implementing parts of the pipeline and the evaluation "
  "scripts. I designed the evaluation, wrote and verified the questions against the filings, ran every experiment on "
  "my machine, and read and labelled every lost answer. The model, embeddings and evaluation all run locally.")

h1("Reproducibility")
p(f"Code, questions, results and the full log: {REPO}.")
code("""
cd rag
python3 ingest.py data/MSFT_10K.pdf --collection msft_10k          # once per filing
python3 eval.py --model ../llama-mlx-quantized --collection msft_10k \\
    --questions eval_questions_MSFT -k 5 --out eval_results_msft.json
python3 eval_oracle.py --collection msft_10k --questions eval_questions_MSFT \\
    --compare eval_results_msft.json --out eval_oracle_msft.json   # gold-chunk oracle
python3 report/make_figures.py && python3 report/build_pdf.py      # this report
""")

def glue_headings(flow: list) -> list:
    """Before each heading, break the page only if the heading and the element after it don't fit."""
    import io

    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import CondPageBreak

    canv = Canvas(io.BytesIO())
    width, max_h = letter[0] - 2 * inch, letter[1] - 1.8 * inch
    def height(fl) -> float:
        if hasattr(fl, "_content"):  # KeepTogether reports a huge height on purpose; measure its parts
            return sum(height(c) for c in fl._content)
        h = fl.wrapOn(canv, width, max_h)[1]
        style = getattr(fl, "style", None)
        return h + (getattr(style, "spaceBefore", 0) + getattr(style, "spaceAfter", 0) if style else 0)

    is_heading = lambda fl: isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2")  # noqa: E731
    out = []
    for i, f in enumerate(flow):
        if is_heading(f) and (i == 0 or not is_heading(flow[i - 1])):
            j, need = i, 0.0
            while j < len(flow) and is_heading(flow[j]):  # the heading run ...
                need += height(flow[j])
                j += 1
            if j < len(flow):  # ... plus the first element after it
                need += height(flow[j])
            out.append(CondPageBreak(min(need + 4, max_h)))
        out.append(f)
    return out


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=1 * inch, rightMargin=1 * inch, topMargin=0.9 * inch,
                        bottomMargin=0.9 * inch, title="Local RAG over SEC 10-K Filings", author=AUTHOR)
doc.build(glue_headings(story), onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}")
