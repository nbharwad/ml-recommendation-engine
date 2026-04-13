"""Retrieval Service gRPC client with Milvus ANN."""

import struct
from typing import Any, Optional

from .base import BaseGRPCClient

# Milvus integration - will be None if pymilvus not available
try:
    from pymilvus import connections, Collection

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class RetrievalClient(BaseGRPCClient):
    """Client for Retrieval Service (Milvus-backed ANN)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50052,
        timeout: float = 1.0,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
    ):
        super().__init__(
            service_name="retrieval",
            host=host,
            port=port,
            timeout=timeout,
        )
        # Wire gRPC stub
        from recommendation.v1 import recommendation_pb2_grpc

        self.set_stub_class(recommendation_pb2_grpc.RetrievalServiceStub)
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self._milvus_conn = None
        self._collection = None

    async def connect(self):
        await super().connect()
        if MILVUS_AVAILABLE:
            try:
                connections.connect("default", host=self.milvus_host, port=self.milvus_port)
                self._collection = Collection("item_embeddings")
                self._collection.load()
            except Exception as e:
                pass  # Fallback to gRPC

    async def retrieve_candidates(
        self,
        user_embedding: list[float],
        num_candidates: int = 100,
        exclude_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidates using ANN search."""
        # Try Milvus first
        if self._collection and MILVUS_AVAILABLE:
            try:
                emb_bytes = struct.pack(f"{len(user_embedding)}f", *user_embedding)
                results = self._collection.search(
                    data=[user_embedding],
                    anns_field="embedding",
                    param={"ef": 64},
                    limit=num_candidates,
                )
                return [
                    {"item_id": h.id, "retrieval_score": 1.0 - h.distance, "source": "ANN"}
                    for h in results[0]
                ]
            except Exception:
                pass  # Fallback

        # Fallback to gRPC/Mock
        if not self._stub:
            return self._fallback_candidates(num_candidates)

        try:
            from recommendation.v1.recommendation_pb2 import RetrievalRequest, PageContext

            emb_bytes = struct.pack(f"{len(user_embedding)}f", *user_embedding)
            req = RetrievalRequest(
                user_embedding=emb_bytes,
                num_candidates=num_candidates,
                exclude_item_ids=exclude_ids or [],
                page_context=PageContext.PAGE_CONTEXT_HOME,
            )
            resp = await self.call("RetrieveCandidates", req)
            return [
                {"item_id": c.item_id, "retrieval_score": c.retrieval_score, "source": c.source}
                for c in resp.candidates
            ]
        except Exception:
            return self._fallback_candidates(num_candidates)

    def _fallback_candidates(self, num: int) -> list[dict]:
        sources = ["ANN", "CF", "TRENDING", "POPULARITY"]
        return [
            {
                "item_id": f"item_{i:07d}",
                "retrieval_score": 1.0 - i * 0.001,
                "source": sources[i % 4],
            }
            for i in range(min(num, 1000))
        ]
