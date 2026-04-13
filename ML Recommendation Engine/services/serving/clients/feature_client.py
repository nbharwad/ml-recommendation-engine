"""Feature Store gRPC client."""

import struct
from typing import Any, Optional

from .base import BaseGRPCClient


class FeatureStoreClient(BaseGRPCClient):
    """Client for Feature Service (gRPC)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        timeout: float = 1.0,
    ):
        super().__init__(
            service_name="feature_store",
            host=host,
            port=port,
            timeout=timeout,
        )
        # Wire gRPC stub
        from recommendation.v1 import recommendation_pb2_grpc

        self.set_stub_class(recommendation_pb2_grpc.FeatureServiceStub)

    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Fetch user features from Feature Service."""
        if not self._stub:
            return self._fallback_user(user_id)

        try:
            from recommendation.v1.recommendation_pb2 import GetUserFeaturesRequest

            req = GetUserFeaturesRequest(user_id=user_id)
            resp = await self.call("GetUserFeatures", req)
            return self._parse_user(resp)
        except Exception as e:
            return self._fallback_user(user_id)

    async def get_batch_item_features(self, item_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch batch item features."""
        if not self._stub:
            return {iid: self._fallback_item(iid) for iid in item_ids}

        try:
            from recommendation.v1.recommendation_pb2 import GetBatchItemFeaturesRequest

            req = GetBatchItemFeaturesRequest(item_ids=item_ids)
            resp = await self.call("GetBatchItemFeatures", req)
            return {v.entity_id: self._parse_item(v) for v in resp.vectors}
        except Exception:
            return {iid: self._fallback_item(iid) for iid in item_ids}

    def _parse_user(self, resp) -> dict:
        features = {}
        for k, v in resp.features.items():
            if v.HasField("float_val"):
                features[k] = v.float_val
            elif v.HasField("int_val"):
                features[k] = v.int_val
            elif v.HasField("string_val"):
                features[k] = v.string_val
        emb = features.pop("embedding", [0.0] * 128)
        return {"user_id": resp.entity_id, "features": features, "embedding": emb}

    def _parse_item(self, resp) -> dict:
        features = {}
        for k, v in resp.features.items():
            if v.HasField("float_val"):
                features[k] = v.float_val
            elif v.HasField("int_val"):
                features[k] = v.int_val
            elif v.HasField("string_val"):
                features[k] = v.string_val
        return {"item_id": resp.entity_id, "features": features}

    def _fallback_user(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "features": {
                "purchase_count_30d": 0,
                "avg_order_value": 0.0,
                "preferred_categories": [],
                "session_click_count": 0,
                "user_segment": "default",
            },
            "embedding": [0.0] * 128,
        }

    def _fallback_item(self, item_id: str) -> dict:
        return {
            "item_id": item_id,
            "features": {"price": 0.0, "ctr_7d": 0.01, "stock_count": 0},
        }
