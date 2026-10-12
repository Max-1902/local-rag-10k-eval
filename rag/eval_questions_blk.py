"""Ground-truth questions for the BlackRock FY2025 10-K.

BlackRock reports two different net income figures: consolidated net income
($5,942M, including non-controlling interests) and net income attributable to
BlackRock, Inc. ($5,553M). q3 and q4 target each one.
"""

EVAL_SET = [
    # --- narrative / qualitative ---
    {
        "id": "n1",
        "type": "narrative",
        "question": "What regulatory risks does BlackRock face in its China operations?",
        "source_page": 18,
        "answer_contains": ["China", "regulatory", "data security"],
    },
    {
        "id": "n2",
        "type": "narrative",
        "question": "What is BlackRock's commission file number and when was its fiscal year 2025 annual report period?",
        "source_page": 1,
        "answer_contains": ["001-42297", "December 31, 2025"],
    },
    {
        "id": "n3",
        "type": "narrative",
        "question": "What private markets data company did BlackRock acquire in March 2025, and for approximately how much?",
        "source_page": 12,
        "answer_contains": ["Preqin Holding Limited", "approximately $3.2 billion", "private markets data"],
    },
    {
        "id": "n4",
        "type": "narrative",
        "question": "What does BlackRock say about being the target of cyber-attacks and impersonation of the company?",
        "source_page": 25,
        "answer_contains": ["target of cyber-attacks", "fraudulent impersonations", "co-opting of its brand"],
    },
    {
        "id": "n5",
        "type": "narrative",
        "question": "What is the common ownership theory that BlackRock has faced commentary about, and which U.S. regulators have examined it?",
        "source_page": 35,
        "answer_contains": ["common ownership theory", "FTC", "anticompetitive effects"],
    },
    {
        "id": "n6",
        "type": "narrative",
        "question": "How much AUM did BlackRock have at December 31, 2025, and approximately how many employees did it have across how many countries?",
        "source_page": 40,
        # page 40 text: "approximately 24,900 employees in more than 30 countries"
        # (30 is the employee-countries figure; "100 countries" in the same
        # sentence refers to client countries, not employees - was previously
        # mismatched to the wrong number)
        "answer_contains": ["$14.0 trillion", "24,900 employees", "30 countries"],
    },
    # --- numeric / tabular ---
    {
        "id": "q1",
        "type": "numeric",
        "question": "What was BlackRock's total revenue for fiscal year 2025?",
        "source_page": [4, 80],
        "answer_contains": ["24,216"],
    },
    {
        "id": "q2",
        "type": "numeric",
        "question": "What was BlackRock's operating income for fiscal year 2025?",
        "source_page": [4, 80],
        "answer_contains": ["7,045"],
    },
    {
        "id": "q3",
        "type": "numeric",
        "question": "What was BlackRock's consolidated net income (before attribution to non-controlling interests) for fiscal year 2025, per the statement of comprehensive income?",
        "source_page": 81,
        "answer_contains": ["5,942"],
    },
    {
        "id": "q4",
        "type": "numeric",
        "question": "What was BlackRock's net income attributable to BlackRock, Inc. for fiscal year 2025?",
        "source_page": 4,
        "answer_contains": ["5,553"],
    },
    {
        "id": "q5",
        "type": "numeric",
        "question": "What was BlackRock's diluted earnings per common share for fiscal year 2025?",
        "source_page": 4,
        "answer_contains": ["35.31"],
    },
    {
        "id": "q6",
        "type": "numeric",
        "question": "What was BlackRock's employee compensation and benefits expense for fiscal year 2025?",
        "source_page": [54, 80],
        "answer_contains": ["8,446"],
    },
    # --- reasoning (requires computing from retrieved figures, not a single lookup) ---
    {
        "id": "r1",
        "type": "reasoning",
        "question": "What was BlackRock's net margin — net income attributable to BlackRock, Inc. as a percentage of total revenue — for fiscal year 2025?",
        "source_page": 4,
        "answer_contains_any": [["22.9%"], ["22.93%"], ["23%"], ["23.0%"]],
    },
]
