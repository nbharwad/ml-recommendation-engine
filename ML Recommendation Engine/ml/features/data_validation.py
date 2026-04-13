"""
Data Validation Pipeline
==========================
Great Expectations-style data validation for training data quality.

Runs before every model training to ensure:
1. Schema integrity (columns, types, nulls)
2. Distribution stability (no unexpected shifts)
3. Volume checks (data completeness)
4. Value range validation
5. Referential integrity

If validation fails → training is BLOCKED and alert is fired.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    description: str
    severity: str  # "critical" (blocks training) or "warning" (alert only)
    
    def validate(self, data: Any) -> tuple[bool, str]:
        """Override in subclasses."""
        raise NotImplementedError


class NullRateRule(ValidationRule):
    """Check null/missing rate for a column."""
    def __init__(self, column: str, max_null_rate: float = 0.001, severity: str = "critical"):
        super().__init__(
            name=f"null_rate_{column}",
            description=f"{column} null rate must be < {max_null_rate*100}%",
            severity=severity,
        )
        self.column = column
        self.max_null_rate = max_null_rate
    
    def validate(self, data: dict[str, np.ndarray]) -> tuple[bool, str]:
        if self.column not in data:
            return False, f"Column {self.column} not found"
        values = data[self.column]
        nulls = np.isnan(values).sum() if values.dtype.kind == 'f' else (values == None).sum()
        null_rate = nulls / max(len(values), 1)
        passed = null_rate <= self.max_null_rate
        msg = f"null_rate={null_rate:.4f}, threshold={self.max_null_rate}"
        return passed, msg


class ValueRangeRule(ValidationRule):
    """Check that values fall within expected range."""
    def __init__(self, column: str, min_val: float = None, max_val: float = None, severity: str = "critical"):
        super().__init__(
            name=f"range_{column}",
            description=f"{column} values must be in [{min_val}, {max_val}]",
            severity=severity,
        )
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, data: dict[str, np.ndarray]) -> tuple[bool, str]:
        if self.column not in data:
            return False, f"Column {self.column} not found"
        values = data[self.column]
        violations = 0
        if self.min_val is not None:
            violations += (values < self.min_val).sum()
        if self.max_val is not None:
            violations += (values > self.max_val).sum()
        passed = violations == 0
        msg = f"violations={violations}/{len(values)}"
        return passed, msg


class VolumeRule(ValidationRule):
    """Check data volume is within expected range."""
    def __init__(self, min_rows: int, max_rows: int = None, severity: str = "critical"):
        super().__init__(
            name="volume_check",
            description=f"Row count must be >= {min_rows}",
            severity=severity,
        )
        self.min_rows = min_rows
        self.max_rows = max_rows
    
    def validate(self, data: dict[str, np.ndarray]) -> tuple[bool, str]:
        row_count = len(next(iter(data.values()))) if data else 0
        passed = row_count >= self.min_rows
        if self.max_rows:
            passed = passed and row_count <= self.max_rows
        msg = f"rows={row_count}, min={self.min_rows}"
        return passed, msg


class DistributionStabilityRule(ValidationRule):
    """
    Population Stability Index (PSI) to detect distribution shifts.
    
    PSI > 0.1: notable shift (warning)
    PSI > 0.25: significant shift (critical — potential data issue)
    """
    def __init__(self, column: str, reference_distribution: np.ndarray, max_psi: float = 0.25, severity: str = "critical"):
        super().__init__(
            name=f"psi_{column}",
            description=f"{column} PSI must be < {max_psi}",
            severity=severity,
        )
        self.column = column
        self.reference_distribution = reference_distribution
        self.max_psi = max_psi
    
    def validate(self, data: dict[str, np.ndarray]) -> tuple[bool, str]:
        if self.column not in data:
            return False, f"Column {self.column} not found"
        
        current = data[self.column]
        psi = self._compute_psi(self.reference_distribution, current)
        passed = psi <= self.max_psi
        msg = f"PSI={psi:.4f}, threshold={self.max_psi}"
        return passed, msg
    
    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Compute Population Stability Index."""
        # Bin both distributions
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Normalize to proportions
        expected_pct = expected_counts / max(expected_counts.sum(), 1)
        actual_pct = actual_counts / max(actual_counts.sum(), 1)
        
        # Avoid division by zero
        expected_pct = np.maximum(expected_pct, 0.0001)
        actual_pct = np.maximum(actual_pct, 0.0001)
        
        psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
        return float(psi)


class LabelDistributionRule(ValidationRule):
    """Check positive label rate is within expected bounds."""
    def __init__(self, label_column: str, expected_rate: float, tolerance: float = 0.2, severity: str = "warning"):
        super().__init__(
            name=f"label_rate_{label_column}",
            description=f"Positive rate should be ~{expected_rate:.3f} ±{tolerance*100}%",
            severity=severity,
        )
        self.label_column = label_column
        self.expected_rate = expected_rate
        self.tolerance = tolerance
    
    def validate(self, data: dict[str, np.ndarray]) -> tuple[bool, str]:
        if self.label_column not in data:
            return False, f"Column {self.label_column} not found"
        
        labels = data[self.label_column]
        actual_rate = labels.mean()
        deviation = abs(actual_rate - self.expected_rate) / max(self.expected_rate, 1e-6)
        passed = deviation <= self.tolerance
        msg = f"actual_rate={actual_rate:.4f}, expected={self.expected_rate:.4f}, deviation={deviation:.2%}"
        return passed, msg


# ---------------------------------------------------------------------------
# Validation Suite
# ---------------------------------------------------------------------------

class DataValidationSuite:
    """
    Pre-training data validation suite.
    
    Typical rules for recommendation training data:
    - user_id: no nulls
    - item_id: no nulls
    - event_type: valid enum values
    - timestamp: within expected range
    - price: positive values
    - features: within expected distributions
    """
    
    def __init__(self):
        self.rules: list[ValidationRule] = []
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: ValidationRule):
        self.rules.append(rule)
        return self
    
    def add_standard_rules(self):
        """Add standard validation rules for recommendation training data."""
        self.rules.extend([
            NullRateRule("user_id", max_null_rate=0.0, severity="critical"),
            NullRateRule("item_id", max_null_rate=0.0, severity="critical"),
            NullRateRule("label", max_null_rate=0.0, severity="critical"),
            NullRateRule("price", max_null_rate=0.01, severity="warning"),
            
            ValueRangeRule("price", min_val=0.0, severity="critical"),
            ValueRangeRule("label", min_val=0.0, max_val=1.0, severity="critical"),
            ValueRangeRule("ctr_7d", min_val=0.0, max_val=1.0, severity="critical"),
            
            VolumeRule(min_rows=100_000, severity="critical"),
            
            LabelDistributionRule("label", expected_rate=0.03, tolerance=0.5, severity="warning"),
        ])
        return self
    
    def validate(self, data: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Run all validation rules.
        
        Returns:
            {
                "passed": bool (all critical rules passed),
                "results": [
                    {"name": "...", "passed": bool, "severity": "...", "message": "..."},
                    ...
                ],
                "critical_failures": int,
                "warnings": int,
            }
        """
        results = []
        critical_failures = 0
        warnings = 0
        
        for rule in self.rules:
            try:
                passed, message = rule.validate(data)
            except Exception as e:
                passed = False
                message = f"Validation error: {str(e)}"
            
            result = {
                "name": rule.name,
                "passed": passed,
                "severity": rule.severity,
                "message": message,
                "description": rule.description,
            }
            results.append(result)
            
            if not passed:
                if rule.severity == "critical":
                    critical_failures += 1
                    self.logger.error(f"❌ CRITICAL: {rule.name} — {message}")
                else:
                    warnings += 1
                    self.logger.warning(f"⚠️ WARNING: {rule.name} — {message}")
            else:
                self.logger.info(f"✅ {rule.name} — {message}")
        
        overall_passed = critical_failures == 0
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Data Validation: {'PASSED' if overall_passed else 'FAILED'}")
        self.logger.info(f"Critical failures: {critical_failures}")
        self.logger.info(f"Warnings: {warnings}")
        self.logger.info(f"{'='*50}")
        
        return {
            "passed": overall_passed,
            "results": results,
            "critical_failures": critical_failures,
            "warnings": warnings,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example validation
    suite = DataValidationSuite().add_standard_rules()
    
    # Synthetic data
    np.random.seed(42)
    n = 500_000
    data = {
        "user_id": np.arange(n, dtype=np.float64),
        "item_id": np.arange(n, dtype=np.float64),
        "label": np.random.binomial(1, 0.03, n).astype(np.float64),
        "price": np.random.uniform(1, 200, n),
        "ctr_7d": np.random.uniform(0, 0.1, n),
    }
    
    results = suite.validate(data)
    print(json.dumps({k: v for k, v in results.items() if k != "results"}, indent=2))
