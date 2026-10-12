"""Ground-truth questions for the Lockheed Martin FY2025 10-K."""

EVAL_SET = [
    # --- narrative / qualitative ---
    {
        "id": "n1",
        "type": "narrative",
        "question": "What happened to the IWSS and C6ISR lines of business within RMS effective January 2026?",
        "source_page": 5,
        "answer_contains": ["restructured", "renamed", "Sensors, Effectors"],
    },
    {
        "id": "n2",
        "type": "narrative",
        "question": "What is an Undefinitized Contract Action (UCA)?",
        "source_page": 12,
        "answer_contains": ["UCA", "U.S. Government", "before the terms"],
    },
    {
        "id": "n3",
        "type": "narrative",
        "question": "What does Lockheed Martin say about pension funding requirements and costs?",
        # page 20 (risk factors) frames the topic abstractly; pages 47 and 85
        # (MD&A / notes) independently restate the same topic with the actual
        # funding figures ("$860 million of cash contributions... in 2025" /
        # "no expected required contributions... in 2026") - verified verbatim
        # on all three pages, not a retrieval failure
        "source_page": [20, 47, 85],
        "answer_contains_any": [
            ["pension", "return on pension assets", "actuarial"],
            ["860 million", "ERISA"],
        ],
    },
    {
        "id": "n4",
        "type": "narrative",
        "question": "What global security conflicts or tensions does Lockheed Martin mention?",
        "source_page": 30,
        "answer_contains": ["Europe", "Middle East", "Pacific"],
    },
    {
        "id": "n5",
        "type": "narrative",
        "question": "What revenue recognition method does Lockheed Martin use for its long-term fixed price contracts?",
        "source_page": 51,
        "answer_contains": ["percentage-of-completion", "cost-to-cost"],
    },
    {
        "id": "n6",
        "type": "narrative",
        "question": "What does Lockheed Martin say about its exposure to litigation and environmental proceedings in Item 3 Legal Proceedings?",
        "source_page": 25,
        "answer_contains": ["party to litigation", "remote that the outcome", "material adverse effect on the corporation as a whole"],
    },
    # --- numeric / tabular ---
    {
        "id": "q1",
        "type": "numeric",
        "question": "What were Lockheed Martin's total sales for fiscal year 2025?",
        "source_page": 52,
        "answer_contains": ["75,048"],
    },
    {
        "id": "q2",
        "type": "numeric",
        "question": "What were Lockheed Martin's net earnings for fiscal year 2025?",
        # same figure legitimately restated on the earnings statement, comprehensive
        # income statement, and cash flow statement
        "source_page": [52, 53, 55],
        "answer_contains": ["5,017"],
    },
    {
        "id": "q3",
        "type": "numeric",
        "question": "What was Lockheed Martin's revenue from Products for fiscal year 2025?",
        "source_page": 52,
        "answer_contains": ["62,654"],
    },
    {
        "id": "q4",
        "type": "numeric",
        "question": "What was Lockheed Martin's cash and cash equivalents balance at the end of fiscal year 2025?",
        "source_page": 54,
        # model correctly rounded to "$4.1 billion" instead of restating "4,121" -
        # accept either the exact figure or the natural billion-scale rephrasing
        "answer_contains_any": [["4,121"], ["4.1 billion"]],
    },
    {
        "id": "q5",
        "type": "numeric",
        "question": "What was Lockheed Martin's stock-based compensation expense in the cash flow statement for fiscal year 2025?",
        "source_page": 55,
        "answer_contains": ["304"],
    },
    {
        "id": "q6",
        "type": "numeric",
        "question": "What was Lockheed Martin's goodwill balance at the end of fiscal year 2025?",
        "source_page": 54,
        "answer_contains_any": [["11,314"], ["11.3 billion"]],
    },
    # --- reasoning (requires computing from retrieved figures, not a single lookup) ---
    {
        "id": "r1",
        "type": "reasoning",
        "question": "What was Lockheed Martin's net profit margin (net earnings as a percentage of total sales) for fiscal year 2025?",
        "source_page": 52,
        "answer_contains_any": [["6.7%"], ["6.69%"], ["6.68%"]],
    },
]
