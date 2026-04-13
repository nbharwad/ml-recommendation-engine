"""Re-Ranking Service gRPC client."""

from typing import Any

from .base import BaseGRPCClient


class ReRankingClient(BaseGRPCClient):
    """Client for Re-Ranking Service (MMR + business rules)."""

    def __init__(self, host: str = "localhost", port: int = 50054, timeout: float = 0.5):
        super().__init__(
            service_name="reranking",
            host=host,
            port=port,
            timeout=timeout,
        )
        # Wire gRPC stub
        from recommendation.v1 import recommendation_pb2_grpc

        self.set_stub_class(recommendation_pb2_grpc.ReRankingServiceStub)

    async def rerank(
        self,
        ranked_items: list[dict[str, Any]],
        diversity_lambda: float = 0.7,
        max_same_category: int = 3,
        output_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Apply MMR diversity + business rules."""
        if not self._stub:
            return self._fallback_rerank(ranked_items, output_size)

        try:
            from recommendation.v1.recommendation_pb2 import ReRankRequest, ReRankingConfig

            req = ReRankRequest(
                config=ReRankingConfig(
                    diversity_lambda=diversity_lambda,
                    max_same_category=max_same_category,
                    output_size=output_size,
                )
            )
            resp = await self.call("ReRank", req)
            return [
                {
                    "item_id": i.item_id,
                    "position": i.position,
                    "final_score": i.final_score,
                    "relevance_score": i.relevance_score,
                    "diversity_score": i.diversity_score,
                }
                for i in resp.items[:output_size]
            ]
        except Exception:
            return self._fallback_rerank(ranked_items, output_size)

    def _fallback_rerank(self, items: list[dict], output_size: int) -> list[dict]:
        return [
            {
                "item_id": it["item_id"],
                "position": i + 1,
                "final_score": it.get("score", 0) * (1 - 0.01 * i),
                "relevance_score": it.get("score", 0),
                "diversity_score": 0.8 - i * 0.02,
            }
            for i, it in enumerate(items[:output_size])
        ]
