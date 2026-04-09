"""Golden dataset for evaluation - 30 test cases across 4 categories."""

from typing import Optional


class TestCase:
    """Single test case in the golden dataset."""

    def __init__(
        self,
        id: str,
        category: str,
        query: str,
        expected_answer_keywords: list[str],
        expected_citations_contain: Optional[list[str]] = None,
        expected_tool_used: Optional[str] = None,
        expected_execution_plan: Optional[str] = None,
        max_acceptable_latency_s: float = 5.0,
    ):
        self.id = id
        self.category = category
        self.query = query
        self.expected_answer_keywords = expected_answer_keywords
        self.expected_citations_contain = expected_citations_contain or []
        self.expected_tool_used = expected_tool_used
        self.expected_execution_plan = expected_execution_plan
        self.max_acceptable_latency_s = max_acceptable_latency_s

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "expected_answer_keywords": self.expected_answer_keywords,
            "expected_citations_contain": self.expected_citations_contain,
            "expected_tool_used": self.expected_tool_used,
            "expected_execution_plan": self.expected_execution_plan,
            "max_acceptable_latency_s": self.max_acceptable_latency_s,
        }


# =============================================================================
# GOLDEN DATASET - 30 TEST CASES
# =============================================================================

GOLDEN_DATASET: list[TestCase] = [
    # =========================================================================
    # Category: Denial Resolution (10 cases)
    # =========================================================================
    TestCase(
        id="DR-001",
        category="denial_resolution",
        query="Why was my claim denied with code CO-4?",
        expected_answer_keywords=["coding error", "procedure code", "modifier"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="DR-002",
        category="denial_resolution",
        query="What does denial code PR-96 mean?",
        expected_answer_keywords=["patient responsibility", "deductible", "coinsurance"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="DR-003",
        category="denial_resolution",
        query="Explain denial code CO-29 and what I should do next",
        expected_answer_keywords=["timely filing", "claim submission deadline", "120 days"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="DR-004",
        category="denial_resolution",
        query="My claim was denied with CO-97. What are my appeal options?",
        expected_answer_keywords=["appeal", "payment adjustment", "contractual"],
        expected_citations_contain=["CARC", "Medicare"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="DR-005",
        category="denial_resolution",
        query="Why did Medicare deny my claim with code CO-16?",
        expected_answer_keywords=["duplicate", "original claim", "timely filing"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="retrieval_only",
    ),
    TestCase(
        id="DR-006",
        category="denial_resolution",
        query="What does OA-23 mean for my commercial insurance claim?",
        expected_answer_keywords=["other adjustment", "deductible", "patient pay"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="DR-007",
        category="denial_resolution",
        query="Claim denied with CO-50. What are the common causes?",
        expected_answer_keywords=["non-covered service", "benefit limitation", "experimental"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="DR-008",
        category="denial_resolution",
        query="How do I appeal a medical necessity denial for code CO-45?",
        expected_answer_keywords=["medical necessity", "clinical documentation", "appeal letter"],
        expected_citations_contain=["CARC", "Medicare"],
        expected_tool_used="appeal_generator",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="DR-009",
        category="denial_resolution",
        query="What is the appeal deadline for denial code PR-2?",
        expected_answer_keywords=["appeal deadline", "30 days", "60 days"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="DR-010",
        category="denial_resolution",
        query="Why was my claim denied with CO-167 and what supporting documentation do I need?",
        expected_answer_keywords=["additional information", "clinical documentation", "missing"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="retrieval_and_tools",
    ),

    # =========================================================================
    # Category: CPT/ICD Coding (8 cases)
    # =========================================================================
    TestCase(
        id="CPT-001",
        category="cpt_coding",
        query="What is the correct CPT code for laparoscopic cholecystectomy?",
        expected_answer_keywords=["47562", "laparoscopic", "cholecystectomy"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="CPT-002",
        category="cpt_coding",
        query="Look up CPT code 99213",
        expected_answer_keywords=["office visit", "established patient", "15 minutes"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="CPT-003",
        category="cpt_coding",
        query="What is the RVU for CPT code 27447?",
        expected_answer_keywords=["total RVU", "work RVU", "knee replacement"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="CPT-004",
        category="cpt_coding",
        query="What modifier should I use for a professional component of a radiology service?",
        expected_answer_keywords=["modifier 26", "professional component", "radiology"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="CPT-005",
        category="cpt_coding",
        query="What is the global period for CPT 36415?",
        expected_answer_keywords=["global period", "0 days", "venipuncture"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="CPT-006",
        category="cpt_coding",
        query="Does CPT 70553 require prior authorization?",
        expected_answer_keywords=["MRI", "brain", "prior authorization"],
        expected_citations_contain=["payer"],
        expected_tool_used="policy_fetcher",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="CPT-007",
        category="cpt_coding",
        query="What are common denial reasons for CPT 80053?",
        expected_answer_keywords=["metabolic panel", "laboratory", "bundling"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),
    TestCase(
        id="CPT-008",
        category="cpt_coding",
        query="What is the difference between CPT 99213 and 99214?",
        expected_answer_keywords=["99214", "25 minutes", "moderate complexity"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="tool_only",
    ),

    # =========================================================================
    # Category: Multi-hop Reasoning (7 cases)
    # =========================================================================
    TestCase(
        id="MH-001",
        category="multi_hop",
        query="Why was my claim for CPT 27447 denied with CO-97 and what does UnitedHealthcare's policy say about knee replacements?",
        expected_answer_keywords=["denial reason", "policy", "coverage criteria"],
        expected_citations_contain=["CARC", "payer"],
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="MH-002",
        category="multi_hop",
        query="What's the appeal deadline for a Medicare denial with code CO-4, and what should I include in my appeal letter?",
        expected_answer_keywords=["120 days", "appeal letter", "clinical justification"],
        expected_citations_contain=["Medicare", "CMS"],
        expected_tool_used="appeal_generator",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="MH-003",
        category="multi_hop",
        query="My claim for CPT 99213 was denied. What does the payer policy say about office visit coding, and what's the typical RVU for this code?",
        expected_answer_keywords=["policy", "RVU", "coding guidelines"],
        expected_citations_contain=["payer"],
        expected_tool_used="cpt_lookup",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="MH-004",
        category="multi_hop",
        query="Explain denial code CO-45, then look up CPT 43239 to see if there's a bundling issue with the procedure I performed.",
        expected_answer_keywords=["medical necessity", "ERCP", "bundling"],
        expected_citations_contain=["CARC"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="MH-005",
        category="multi_hop",
        query="What does Aetna's policy say about prior authorization for MRIs, and what's the typical denial reason for CPT 70553?",
        expected_answer_keywords=["prior auth", "MRI", "coverage"],
        expected_citations_contain=["Aetna"],
        expected_tool_used="policy_fetcher",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="MH-006",
        category="multi_hop",
        query="I received denial CO-29 for a claim filed 95 days after service. What are my options, and what's the correct way to file a timely appeal?",
        expected_answer_keywords=["timely filing", "appeal", "documentation"],
        expected_citations_contain=["Medicare"],
        expected_tool_used="denial_explainer",
        expected_execution_plan="retrieval_and_tools",
    ),
    TestCase(
        id="MH-007",
        category="multi_hop",
        query="Find the Medicare policy for CPT 74177 (abdomen CT), check if it requires prior auth, and explain what denial code is common for this type of service.",
        expected_answer_keywords=["CT abdomen", "prior authorization", "denial"],
        expected_citations_contain=["Medicare"],
        expected_execution_plan="retrieval_and_tools",
    ),

    # =========================================================================
    # Category: Ambiguous / Edge Cases (5 cases)
    # =========================================================================
    TestCase(
        id="AE-001",
        category="ambiguous",
        query="Help",
        expected_answer_keywords=[],
        expected_execution_plan="clarification_needed",
    ),
    TestCase(
        id="AE-002",
        category="ambiguous",
        query="What do I do about my denied claim?",
        expected_answer_keywords=[],
        expected_execution_plan="clarification_needed",
    ),
    TestCase(
        id="AE-003",
        category="ambiguous",
        query="billing question",
        expected_answer_keywords=[],
        expected_execution_plan="clarification_needed",
    ),
    TestCase(
        id="AE-004",
        category="ambiguous",
        query="Why was my claim denied?",
        expected_answer_keywords=["denial code", "specific"],
        expected_execution_plan="retrieval_only",
    ),
    TestCase(
        id="AE-005",
        category="ambiguous",
        query="How much will insurance pay?",
        expected_answer_keywords=["coverage", "specific CPT"],
        expected_execution_plan="clarification_needed",
    ),
]


def get_all_test_cases() -> list[TestCase]:
    """Return all test cases in the dataset."""
    return GOLDEN_DATASET


def get_test_cases_by_category(category: str) -> list[TestCase]:
    """Return test cases filtered by category.
    
    Args:
        category: Category name (denial_resolution, cpt_coding, multi_hop, ambiguous)
    
    Returns:
        List of test cases in that category
    """
    return [tc for tc in GOLDEN_DATASET if tc.category == category]


def get_test_case_by_id(test_id: str) -> Optional[TestCase]:
    """Get a specific test case by ID.
    
    Args:
        test_id: Test case ID (e.g., "DR-001")
    
    Returns:
        TestCase if found, None otherwise
    """
    for tc in GOLDEN_DATASET:
        if tc.id == test_id:
            return tc
    return None


def get_category_counts() -> dict[str, int]:
    """Get count of test cases per category."""
    counts: dict[str, int] = {}
    for tc in GOLDEN_DATASET:
        counts[tc.category] = counts.get(tc.category, 0) + 1
    return counts


def to_json() -> list[dict]:
    """Convert dataset to JSON-serializable format."""
    return [tc.to_dict() for tc in GOLDEN_DATASET]