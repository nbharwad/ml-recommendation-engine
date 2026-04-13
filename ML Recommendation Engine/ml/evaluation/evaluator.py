"""
Offline Model Evaluation Pipeline
===================================
Comprehensive offline evaluation of recommendation models
before deployment to production.

Evaluates:
1. Ranking quality: AUC, log-loss, NDCG
2. Retrieval quality: Recall@K, Hit Rate
3. Calibration: expected calibration error
4. Business metrics: predicted CTR, coverage, diversity
5. Fairness: performance across user segments

Gates:
- AUC must exceed current production model
- Lift > 0% vs production
- No regression on any user segment by >5%
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    # Quality gates (must pass to deploy)
    min_auc: float = 0.70
    min_auc_lift_vs_baseline: float = 0.0  # must be positive lift
    max_calibration_error: float = 0.05
    min_recall_at_100: float = 0.20
    max_segment_regression: float = 0.05   # no segment can regress by >5%
    
    # Evaluation parameters
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    user_segments: list[str] = field(default_factory=lambda: [
        "high_value", "medium_value", "low_value", "new_user", "returning",
    ])


# ---------------------------------------------------------------------------
# Ranking Metrics
# ---------------------------------------------------------------------------

class RankingMetrics:
    """Offline ranking quality metrics."""
    
    @staticmethod
    def auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Area Under ROC Curve."""
        # Sort by scores descending
        sorted_indices = np.argsort(-scores)
        labels_sorted = labels[sorted_indices]
        
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Count correct orderings
        cum_pos = np.cumsum(labels_sorted)
        auc = (cum_pos * (1 - labels_sorted)).sum() / (n_pos * n_neg)
        
        return float(1 - auc)  # correction for descending sort
    
    @staticmethod
    def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-7
        probs = np.clip(probabilities, eps, 1 - eps)
        loss = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        return float(loss.mean())
    
    @staticmethod
    def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        sorted_indices = np.argsort(-scores)[:k]
        dcg = sum(
            labels[idx] / np.log2(pos + 2)
            for pos, idx in enumerate(sorted_indices)
        )
        
        # Ideal DCG
        ideal_indices = np.argsort(-labels)[:k]
        idcg = sum(
            labels[idx] / np.log2(pos + 2)
            for pos, idx in enumerate(ideal_indices)
        )
        
        return float(dcg / max(idcg, 1e-10))
    
    @staticmethod
    def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
        """Precision at K."""
        sorted_indices = np.argsort(-scores)[:k]
        return float(labels[sorted_indices].mean())
    
    @staticmethod
    def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
        """Recall at K."""
        sorted_indices = np.argsort(-scores)[:k]
        total_relevant = labels.sum()
        if total_relevant == 0:
            return 0.0
        return float(labels[sorted_indices].sum() / total_relevant)


# ---------------------------------------------------------------------------
# Calibration Metrics
# ---------------------------------------------------------------------------

class CalibrationMetrics:
    """Model calibration quality."""
    
    @staticmethod
    def expected_calibration_error(
        labels: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE).
        
        Measures how well-calibrated the model's probabilities are.
        A well-calibrated model: when it predicts 5% click probability,
        approximately 5% of those items should be clicked.
        
        Target: ECE < 0.05 (5%)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_conf = probabilities[mask].mean()
            bin_acc = labels[mask].mean()
            bin_weight = mask.sum() / len(labels)
            
            ece += bin_weight * abs(bin_conf - bin_acc)
        
        return float(ece)
    
    @staticmethod
    def reliability_diagram(
        labels: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10,
    ) -> list[dict[str, float]]:
        """Data for reliability diagram visualization."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []
        
        for i in range(n_bins):
            mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            
            bins.append({
                "bin_center": float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2),
                "predicted_prob": float(probabilities[mask].mean()),
                "actual_prob": float(labels[mask].mean()),
                "count": int(mask.sum()),
            })
        
        return bins


# ---------------------------------------------------------------------------
# Coverage & Diversity Metrics
# ---------------------------------------------------------------------------

class CoverageMetrics:
    """Catalog coverage and recommendation diversity."""
    
    @staticmethod
    def catalog_coverage(
        recommended_items: list[list[str]],
        total_catalog_size: int,
    ) -> float:
        """Fraction of catalog items that appear in any recommendation list."""
        all_items = set()
        for rec_list in recommended_items:
            all_items.update(rec_list)
        return len(all_items) / max(total_catalog_size, 1)
    
    @staticmethod
    def gini_diversity(item_counts: dict[str, int]) -> float:
        """
        Gini coefficient of item exposure distribution.
        0 = perfect equality (all items shown equally)
        1 = perfect inequality (one item gets all exposure)
        
        Target: < 0.8 (avoid popularity bias)
        """
        if not item_counts:
            return 0.0
        
        counts = sorted(item_counts.values())
        n = len(counts)
        total = sum(counts)
        
        if total == 0:
            return 0.0
        
        gini = sum(
            (2 * i - n - 1) * count
            for i, count in enumerate(counts, 1)
        ) / (n * total)
        
        return float(abs(gini))


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    End-to-end model evaluation pipeline.
    
    Runs before every model deployment to ensure quality.
    If any quality gate fails, deployment is blocked.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        labels: np.ndarray,
        scores_candidate: np.ndarray,
        scores_baseline: np.ndarray | None = None,
        user_segments: np.ndarray | None = None,
        item_ids: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run full evaluation suite.
        
        Returns:
            {
                "passed": bool,
                "metrics": {...},
                "gates": {...},
                "segment_metrics": {...},
            }
        """
        results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": {},
            "gates": {},
            "passed": True,
        }
        
        # === Ranking Metrics ===
        metrics = results["metrics"]
        metrics["auc"] = RankingMetrics.auc(labels, scores_candidate)
        metrics["log_loss"] = RankingMetrics.log_loss(labels, scores_candidate)
        
        for k in self.config.k_values:
            metrics[f"ndcg@{k}"] = RankingMetrics.ndcg_at_k(labels, scores_candidate, k)
            metrics[f"precision@{k}"] = RankingMetrics.precision_at_k(labels, scores_candidate, k)
            metrics[f"recall@{k}"] = RankingMetrics.recall_at_k(labels, scores_candidate, k)
        
        # === Calibration ===
        metrics["ece"] = CalibrationMetrics.expected_calibration_error(labels, scores_candidate)
        metrics["reliability_diagram"] = CalibrationMetrics.reliability_diagram(labels, scores_candidate)
        
        # === Baseline Comparison ===
        if scores_baseline is not None:
            baseline_auc = RankingMetrics.auc(labels, scores_baseline)
            metrics["baseline_auc"] = baseline_auc
            metrics["auc_lift"] = metrics["auc"] - baseline_auc
            metrics["relative_auc_lift"] = (metrics["auc"] - baseline_auc) / max(baseline_auc, 1e-6)
        
        # === Quality Gates ===
        gates = results["gates"]
        
        gates["auc_gate"] = {
            "passed": metrics["auc"] >= self.config.min_auc,
            "threshold": self.config.min_auc,
            "value": metrics["auc"],
        }
        
        gates["calibration_gate"] = {
            "passed": metrics["ece"] <= self.config.max_calibration_error,
            "threshold": self.config.max_calibration_error,
            "value": metrics["ece"],
        }
        
        if scores_baseline is not None:
            gates["lift_gate"] = {
                "passed": metrics.get("auc_lift", 0) >= self.config.min_auc_lift_vs_baseline,
                "threshold": self.config.min_auc_lift_vs_baseline,
                "value": metrics.get("auc_lift", 0),
            }
        
        gates["recall_gate"] = {
            "passed": metrics.get("recall@100", 0) >= self.config.min_recall_at_100,
            "threshold": self.config.min_recall_at_100,
            "value": metrics.get("recall@100", 0),
        }
        
        # === Segment Analysis ===
        if user_segments is not None:
            segment_metrics = {}
            for segment in np.unique(user_segments):
                mask = user_segments == segment
                if mask.sum() < 100:
                    continue
                
                seg_auc = RankingMetrics.auc(labels[mask], scores_candidate[mask])
                segment_metrics[str(segment)] = {
                    "auc": seg_auc,
                    "samples": int(mask.sum()),
                    "positive_rate": float(labels[mask].mean()),
                }
                
                # Check for segment regression
                if scores_baseline is not None:
                    seg_baseline_auc = RankingMetrics.auc(labels[mask], scores_baseline[mask])
                    regression = seg_baseline_auc - seg_auc
                    segment_metrics[str(segment)]["regression"] = float(regression)
                    
                    if regression > self.config.max_segment_regression:
                        gates[f"segment_{segment}_gate"] = {
                            "passed": False,
                            "threshold": self.config.max_segment_regression,
                            "value": regression,
                        }
            
            results["segment_metrics"] = segment_metrics
        
        # === Overall Pass/Fail ===
        results["passed"] = all(g["passed"] for g in gates.values())
        
        # Log results
        status = "✅ PASSED" if results["passed"] else "❌ FAILED"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Model Evaluation: {status}")
        self.logger.info(f"AUC: {metrics['auc']:.4f}")
        self.logger.info(f"ECE: {metrics['ece']:.4f}")
        if "auc_lift" in metrics:
            self.logger.info(f"AUC Lift: {metrics['auc_lift']:.4f}")
        for gate_name, gate in gates.items():
            icon = "✅" if gate["passed"] else "❌"
            self.logger.info(f"  {icon} {gate_name}: {gate['value']:.4f} (threshold: {gate['threshold']:.4f})")
        self.logger.info(f"{'='*60}")
        
        return results
    
    def generate_report(self, results: dict[str, Any], output_path: str):
        """Generate evaluation report as JSON."""
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        
        self.logger.info(f"Evaluation report saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument("--predictions", required=True, help="Path to model predictions")
    parser.add_argument("--baseline", help="Path to baseline predictions")
    parser.add_argument("--output", default="reports/evaluation_report.json")
    
    args = parser.parse_args()
    
    config = EvaluationConfig()
    evaluator = ModelEvaluator(config)
    
    # In production: load predictions from Parquet
    print(f"Running model evaluation pipeline")
    print(f"Config: {config}")
    
    # Example with synthetic data
    np.random.seed(42)
    n = 100_000
    labels = np.random.binomial(1, 0.03, n)
    scores = np.random.uniform(0, 0.1, n) + labels * 0.05
    baseline_scores = np.random.uniform(0, 0.1, n) + labels * 0.04
    
    results = evaluator.evaluate(labels, scores, baseline_scores)
    evaluator.generate_report(results, args.output)
