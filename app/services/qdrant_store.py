import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ReferenceVectorRecord:
    point_id: str
    user_id: str
    logo_id: str
    logo_name: str
    reference_image_id: str
    vector: list[float]


class QdrantVectorService:
    def __init__(self) -> None:
        api_key = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None

        # Configure proxy settings for Qdrant
        # When QDRANT_USE_PROXY is False, pass trust_env=False to disable proxy
        # When True, httpx will automatically use HTTP_PROXY/HTTPS_PROXY from environment
        # trust_env is passed through to httpx.Client via **kwargs
        client_kwargs = {"trust_env": settings.QDRANT_USE_PROXY}
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
            check_compatibility=False,
            **client_kwargs,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def _collection_exists(self) -> bool:
        """Check if collection exists by listing all collections."""
        try:
            collections = self.client.get_collections()
            return any(c.name == self.collection_name for c in collections.collections)
        except Exception as exc:
            logger.warning("Collection check failed: %s", exc)
            return False

    def ensure_collection(self, vector_size: int) -> None:
        if not self._collection_exists():
            logger.info("Creating Qdrant collection: %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            return

        collection_info = self.client.get_collection(self.collection_name)
        vectors_config = collection_info.config.params.vectors
        current_size = (
            vectors_config.size if hasattr(vectors_config, "size") else vectors_config["size"]
        )
        if current_size != vector_size:
            raise ValueError(
                f"Existing Qdrant collection dimension {current_size} does not match {vector_size}"
            )

    def upsert_reference_embeddings(self, records: list[ReferenceVectorRecord]) -> None:
        if not records:
            return

        self.ensure_collection(len(records[0].vector))
        points = [
            qdrant_models.PointStruct(
                id=record.point_id,
                vector=record.vector,
                payload={
                    "user_id": record.user_id,
                    "logo_id": record.logo_id,
                    "logo_name": record.logo_name,
                    "reference_image_id": record.reference_image_id,
                    "kind": "reference",
                },
            )
            for record in records
        ]
        self.client.upsert(collection_name=self.collection_name, wait=True, points=points)

    def search_reference_embeddings(
        self,
        user_id: str,
        vector: list[float],
        limit: int,
    ) -> list[Any]:
        if not self._collection_exists():
            return []

        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id",
                    match=qdrant_models.MatchValue(value=user_id),
                ),
                qdrant_models.FieldCondition(
                    key="kind",
                    match=qdrant_models.MatchValue(value="reference"),
                ),
            ]
        )
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return list(search_result.points)
