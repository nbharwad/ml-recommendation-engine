"""Evaluation package - golden dataset, metrics, and evaluation pipeline."""

from app.evaluation.dataset import (
    GOLDEN_DATASET,
    get_all_test_cases,
    get_category_counts,
    get_test_case_by_id,
    get_test_cases_by_category,
    to_json,
)

__all__ = [
    "GOLDEN_DATASET",
    "get_all_test_cases",
    "get_test_cases_by_category",
    "get_test_case_by_id",
    "get_category_counts",
    "to_json",
]