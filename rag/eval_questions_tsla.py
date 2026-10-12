"""Ground-truth questions for the Tesla FY2025 10-K.

`source_page` may list several pages when a figure is restated verbatim in
more than one section (e.g. MD&A and a segment note).
"""

EVAL_SET = [
    # --- narrative / qualitative ---
    {
        "id": "n1",
        "type": "narrative",
        "question": "What AI-related products and services is Tesla focused on bringing into the real world?",
        "source_page": 5,
        "answer_contains": ["Full Self-Driving", "Robotaxi", "Optimus"],
    },
    {
        "id": "n2",
        "type": "narrative",
        "question": "How does Tesla typically achieve vehicle sales without relying on traditional advertising?",
        "source_page": 8,
        "answer_contains": ["advertising", "marketing"],
    },
    {
        "id": "n3",
        "type": "narrative",
        "question": "Who oversees Tesla's human capital management strategy?",
        "source_page": 15,
        "answer_contains": ["Compensation Committee"],
    },
    {
        "id": "n4",
        "type": "narrative",
        "question": "What does Tesla's ability to make scheduled debt payments depend on?",
        "source_page": 25,
        "answer_contains": ["future performance", "economic", "financial"],
    },
    {
        "id": "n5",
        "type": "narrative",
        "question": "What act governs the forward-looking statements in Tesla's 10-K?",
        "source_page": 4,
        "answer_contains": ["Private Securities Litigation Reform Act", "1995"],
    },
    {
        "id": "n6",
        "type": "narrative",
        "question": "In which U.S. states and foreign countries does Tesla have manufacturing facilities, and what new facility began operations there in January 2026?",
        "source_page": 10,
        "answer_contains": ["California, New York, Texas and Nevada", "China and Germany", "lithium refinery in Texas"],
    },
    # --- numeric / tabular ---
    {
        "id": "q1",
        "type": "numeric",
        "question": "What were Tesla's total revenues for fiscal year 2025?",
        # restated in the revenue-recognition note (page 58 states total revenues
        # from sales and services excluding leasing; page 53 is the full total
        # including leasing) - using the income statement's full total as primary
        "source_page": 53,
        "answer_contains": ["94,827"],
    },
    {
        "id": "q2",
        "type": "numeric",
        "question": "What was Tesla's net income for fiscal year 2025 according to the statement of comprehensive income?",
        "source_page": [54, 56],
        "answer_contains": ["3,855"],
    },
    {
        "id": "q3",
        "type": "numeric",
        "question": "What was Tesla's automotive sales revenue for fiscal year 2025?",
        "source_page": [53, 58],
        "answer_contains": ["65,821"],
    },
    {
        "id": "q4",
        "type": "numeric",
        "question": "What was Tesla's stock-based compensation expense in the cash flow statement for fiscal year 2025?",
        "source_page": 56,
        "answer_contains": ["2,825"],
    },
    {
        "id": "q5",
        "type": "numeric",
        "question": "What was Tesla's net income attributable to common stockholders for fiscal year 2025?",
        "source_page": 65,
        "answer_contains": ["3,794"],
    },
    {
        "id": "q6",
        "type": "numeric",
        "question": "What was Tesla's energy generation and storage revenue for fiscal year 2025?",
        # segment note on page 97 states "Energy generation and storage segment Revenues $ 12,771" verbatim
        "source_page": [53, 97],
        "answer_contains": ["12,771"],
    },
    # --- reasoning (requires computing from retrieved figures, not a single lookup) ---
    {
        "id": "r1",
        "type": "reasoning",
        "question": "What percentage of Tesla's total revenues for fiscal year 2025 came from energy generation and storage?",
        "source_page": 53,
        "answer_contains_any": [["13.5%"], ["13.47%"], ["13%"]],
    },
]
