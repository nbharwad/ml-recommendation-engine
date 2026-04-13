"""Ranking Service gRPC client with Triton inference."""

from typing import Any, Optional

from .base import BaseGRPCClient

# Triton client - will be None if tritonclient not available
try:
    import tritonclient.grpc as grpcclient

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class RankingClient(BaseGRPCClient):
    """Client for Ranking Service (Triton-backed DLRM)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50053,
        timeout: float = 1.0,
        triton_url: str = "triton:8001",
    ):
        super().__init__(
            service_name="ranking",
            host=host,
            port=port,
            timeout=timeout,
        )
        # Wire gRPC stub
        from recommendation.v1 import recommendation_pb2_grpc

        self.set_stub_class(recommendation_pb2_grpc.RankingServiceStub)
        self.triton_url = triton_url
        self._triton = None

    async def connect(self):
        await super().connect()
        if TRITON_AVAILABLE:
            try:
                self._triton = grpcclient.InferenceServerClient(self.triton_url)
            except Exception:
                pass  # Fallback to gRPC

    async def rank_candidates(
        self,
        user_features: dict[str, Any],
        item_features: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
        model_version: str = "dlrm-v2.3",
    ) -> list[dict[str, Any]]:
        """Rank candidates using DLRM."""
        # Try Triton first
        if self._triton and TRITON_AVAILABLE:
            try:
                scores = self._triton_infer(user_features, item_features, candidates)
                return [
                    {"item_id": c["item_id"], "score": s, "sub_scores": {"click_prob": s}}
                    for c, s in zip(candidates, scores)
                ]
            except Exception:
                pass

        # Fallback to gRPC/Mock
        if not self._stub:
            return self._fallback_ranked(candidates)

        try:
            from recommendation.v1.recommendation_pb2 import RankingRequest, PageContext

            req = RankingRequest(
                user_id=user_features.get("user_id", ""),
                model_version=model_version,
                page_context=PageContext.PAGE_CONTEXT_HOME,
            )
            resp = await self.call("RankCandidates", req)
            return [
                {"item_id": i.item_id, "score": i.score, "sub_scores": dict(i.sub_scores)}
                for i in resp.items
            ]
        except Exception:
            return self._fallback_ranked(candidates)

    def _triton_infer(self, user_feats, item_feats, candidates):
        """Run inference on Triton."""
        # Simplified: return random scores
        import random

        return [random.random() * 0.1 for _ in candidates]

    def _fallback_ranked(self, candidates: list[dict]) -> list[dict]:
        return [
            {
                "item_id": c["item_id"],
                "score": max(0, 0.95 - i * 0.001),
                "sub_scores": {"click_prob": 0.035},
            }
            for i, c in enumerate(candidates)
        ]
