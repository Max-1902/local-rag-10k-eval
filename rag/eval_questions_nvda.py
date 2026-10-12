"""Ground-truth questions for the NVIDIA FY2026 10-K (fiscal year ended January 25, 2026)."""

EVAL_SET = [
    # --- narrative / qualitative ---
    {
        "id": "n1",
        "type": "narrative",
        "question": "What export restrictions did the U.S. government announce in August 2022 affecting NVIDIA?",
        "source_page": 10,
        "answer_contains": ["export restrictions", "China", "semiconductor"],
    },
    {
        "id": "n2",
        "type": "narrative",
        "question": "Where is NVIDIA headquartered and when was it incorporated?",
        "source_page": 56,
        "answer_contains": ["Santa Clara", "California", "1993"],
    },
    {
        "id": "n3",
        "type": "narrative",
        "question": "What cybersecurity risks does NVIDIA face related to employees and contractors?",
        "source_page": 20,
        "answer_contains": ["cybersecurity", "employees", "contractors"],
    },
    {
        "id": "n4",
        "type": "narrative",
        "question": "What did NVIDIA acquire or license from Groq in December 2025?",
        "source_page": 61,
        "answer_contains": ["Groq", "language processing unit", "license"],
    },
    {
        "id": "n5",
        "type": "narrative",
        "question": "What data privacy and security obligations does NVIDIA say it is subject to?",
        "source_page": 30,
        "answer_contains": ["data privacy", "security laws", "regulations"],
    },
    {
        "id": "n6",
        "type": "narrative",
        "question": "What risk does NVIDIA describe regarding long manufacturing lead times and customer demand estimation?",
        "source_page": 15,
        "answer_contains": ["long manufacturing lead times", "mismatches between supply and demand", "extended lead times of more than 12 months"],
    },
    # --- numeric / tabular ---
    {
        "id": "q1",
        "type": "numeric",
        "question": "What was NVIDIA's total revenue for fiscal year 2026 (year ended January 25, 2026)?",
        # restated verbatim on page 79 (revenue-by-end-market note), not just page 51
        "source_page": [51, 79],
        "answer_contains": ["215,938"],
    },
    {
        "id": "q2",
        "type": "numeric",
        "question": "What was NVIDIA's research and development expense for fiscal year 2026?",
        # MD&A operating expenses table on page 41 states "Research and development $ 18,497" verbatim
        "source_page": [41, 51],
        "answer_contains": ["18,497"],
    },
    {
        "id": "q3",
        "type": "numeric",
        "question": "What was NVIDIA's net income for fiscal year 2026 according to the statement of comprehensive income?",
        "source_page": 52,
        "answer_contains": ["120,067"],
    },
    {
        "id": "q4",
        "type": "numeric",
        "question": "What was NVIDIA's stock-based compensation expense in the cash flow statement for fiscal year 2026?",
        "source_page": 55,
        "answer_contains": ["6,386"],
    },
    {
        "id": "q5",
        "type": "numeric",
        "question": "How much revenue did NVIDIA recognize in fiscal year 2026 that was included in the prior year's deferred revenue balance?",
        "source_page": 69,
        "answer_contains": ["974"],
    },
    {
        "id": "q6",
        "type": "numeric",
        "question": "What was NVIDIA's gross profit for fiscal year 2026?",
        "source_page": 51,
        "answer_contains": ["153,463"],
    },
    # --- reasoning (requires computing from retrieved figures, not a single lookup) ---
    {
        "id": "r1",
        "type": "reasoning",
        "question": "What was NVIDIA's net profit margin (net income as a percentage of revenue) for fiscal year 2026?",
        "source_page": 51,
        "answer_contains_any": [["55.6%"], ["55.6"], ["56%"]],
    },
]
