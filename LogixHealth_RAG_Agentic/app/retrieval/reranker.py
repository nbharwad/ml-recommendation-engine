"""Domain-aware reranker for search results."""

from typing import Optional

from app.retrieval.azure_search import SearchDocument
from app.config import settings


class Reranker:
    """Domain-aware reranker combining multiple relevance signals."""

    def __init__(
        self,
        semantic_weight: Optional[float] = None,
        recency_weight: Optional[float] = None,
        authority_weight: Optional[float] = None,
        exact_match_weight: Optional[float] = None,
    ):
        """Initialize reranker with configurable weights."""
        self.semantic_weight = semantic_weight or settings.reranker_semantic_weight
        self.recency_weight = recency_weight or settings.reranker_recency_weight
        self.authority_weight = authority_weight or settings.reranker_authority_weight
        self.exact_match_weight = exact_match_weight or settings.reranker_exact_match_weight

        total = (
            self.semantic_weight
            + self.recency_weight
            + self.authority_weight
            + self.exact_match_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError("Reranker weights must sum to 1.0")

    def rerank(
        self,
        documents: list[SearchDocument],
        query: str,
        top_k: Optional[int] = None,
    ) -> list[SearchDocument]:
        """
        Rerank documents using weighted combination of signals.

        Args:
            documents: List of documents to rerank
            query: Original query for exact match detection
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        top_k = top_k or settings.retrieval_top_k
        query_lower = query.lower()

        for doc in documents:
            combined_score = self._calculate_combined_score(doc, query_lower)
            doc.metadata["combined_score"] = combined_score

        reranked = sorted(
            documents, key=lambda d: d.metadata.get("combined_score", 0), reverse=True
        )
        return reranked[:top_k]

    def _calculate_combined_score(self, doc: SearchDocument, query: str) -> float:
        """Calculate weighted combination of relevance signals."""
        semantic_score = doc.score

        recency_score = self._calculate_recency_score(doc.last_updated)

        source = doc.metadata.get("source") or doc.metadata.get("publisher")
        authority_score = self._calculate_authority_score(source)

        exact_match_score = self._calculate_exact_match_score(doc, query)

        combined = (
            (self.semantic_weight * semantic_score)
            + (self.recency_weight * recency_score)
            + (self.authority_weight * authority_score)
            + (self.exact_match_weight * exact_match_score)
        )

        return min(1.0, combined)

    def _calculate_recency_score(self, last_updated: Optional[str]) -> float:
        """Calculate recency score based on last updated date (yearly buckets per spec)."""
        if not last_updated:
            return 0.5

        try:
            from datetime import datetime, timezone

            if "Z" in last_updated:
                last_updated = last_updated.replace("Z", "+00:00")

            doc_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)

            years_old = (now - doc_date).days / 365.0

            if years_old < 1:
                return 1.0
            elif years_old < 2:
                return 0.7
            elif years_old < 5:
                return 0.4
            else:
                return 0.1

        except Exception:
            return 0.5

    def _calculate_authority_score(self, source: Optional[str]) -> float:
        """Calculate authority score based on document source."""
        if not source:
            return 0.5

        source_lower = source.lower()

        if source_lower in ("cms", "medicare", "federal", "hhs", "gov"):
            return 1.0
        elif source_lower in ("payer", "insurance", "bcbs", "uhc", "aetna", "cigna"):
            return 0.8
        elif source_lower in ("clinical", "guideline", "aha", "mayo"):
            return 0.5
        else:
            return 0.3

    def _calculate_exact_match_score(self, doc: SearchDocument, query: str) -> float:
        """Calculate exact match score based on metadata entity matching."""
        if not query:
            return 0.0

        import re

        cpt_pattern = r"\b\d{5}\b"
        denial_pattern = r"\b(?:CO|PR|OA|PI|MO)-\d+\b"

        cpt_matches = re.findall(cpt_pattern, query)
        denial_matches = re.findall(denial_pattern, query)

        doc_cpt = doc.cpt_code or doc.metadata.get("cpt_code")
        doc_denial = doc.denial_code or doc.metadata.get("denial_code")

        for cpt in cpt_matches:
            if doc_cpt and cpt == doc_cpt:
                return 1.0

        for denial in denial_matches:
            if doc_denial and denial.upper() == doc_denial.upper():
                return 1.0

        return 0.0


def create_reranker() -> Reranker:
    """Factory function to create reranker from settings."""
    return Reranker()
