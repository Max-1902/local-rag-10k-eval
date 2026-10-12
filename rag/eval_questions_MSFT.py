"""Ground-truth questions for the Microsoft FY2026 10-K."""

EVAL_SET = [
    # --- narrative / qualitative ---
    {
        "id": "n1",
        "type": "narrative",
        "question": "What sustainability goals did Microsoft announce in 2020?",
        "source_page": 5,
        "answer_contains": ["carbon negative", "water positive", "zero waste", "2030"],
    },
    {
        "id": "n2",
        "type": "narrative",
        "question": "What does LinkedIn's Talent Solutions and Marketing Solutions business do?",
        "source_page": 6,
        "answer_contains": ["LinkedIn", "Talent Solutions", "Marketing Solutions"],
    },
    {
        "id": "n3",
        "type": "narrative",
        "question": "What competition does Azure face?",
        "source_page": 7,
        "answer_contains": ["cloud service providers", "open-source", "hybrid cloud"],
    },
    {
        "id": "n4",
        "type": "narrative",
        "question": "What does Microsoft do when it experiences a cybersecurity incident?",
        "source_page": 30,
        "answer_contains": ["incident response", "triaged", "post incident review"],
    },
    {
        "id": "n5",
        "type": "narrative",
        "question": "Where is Microsoft's corporate headquarters and how much space does it occupy?",
        "source_page": 31,
        "answer_contains": ["Redmond", "Washington", "14 million square feet"],
    },
    {
        "id": "n6",
        "type": "narrative",
        "question": "What stock exchange and ticker symbol does Microsoft common stock trade under?",
        "source_page": 32,
        "answer_contains": ["NASDAQ", "MSFT"],
    },
    # --- numeric / tabular ---
    {
        "id": "q1",
        "type": "numeric",
        "question": "How much did Microsoft's research and development expense increase in fiscal year 2026 compared to 2025, in dollars and percentage?",
        "source_page": 40,
        "answer_contains": ["35,562", "32,488", "9%"],
    },
    {
        "id": "q2",
        "type": "numeric",
        "question": "What was Microsoft's total other income (expense), net for fiscal year 2026?",
        "source_page": 41,
        "answer_contains": ["10,697"],
    },
    {
        "id": "q3",
        "type": "numeric",
        "question": "What was Microsoft's net income for fiscal year 2026 according to the cash flow statement?",
        "source_page": 53,
        "answer_contains": ["133,749"],
    },
    {
        "id": "q4",
        "type": "numeric",
        "question": "How much did Microsoft pay in common stock cash dividends during fiscal year 2026?",
        "source_page": 54,
        "answer_contains": ["27,034"],
    },
    {
        "id": "q5",
        "type": "numeric",
        "question": "What were total dividends declared by Microsoft's Board of Directors in fiscal years 2026 and 2025?",
        "source_page": 45,
        "answer_contains": ["27.0 billion", "24.7 billion"],
    },
    {
        "id": "q6",
        "type": "numeric",
        "question": "What was the ending balance of common stock and paid-in capital at the end of fiscal year 2026?",
        "source_page": 54,
        "answer_contains": ["117,406"],
    },
    # --- reasoning (requires computing from retrieved figures, not a single lookup) ---
    {
        "id": "r1",
        "type": "reasoning",
        "question": "What was Microsoft's total operating margin (operating income as a percentage of revenue) for fiscal year 2026, and how did it compare to fiscal year 2025?",
        "source_page": 37,
        "answer_contains_any": [["46.8%", "45.6%"], ["47%", "46%"], ["46.78%", "45.62%"]],
    },
]
