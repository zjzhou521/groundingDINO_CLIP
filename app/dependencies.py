from functools import lru_cache

from app.core.config import settings
from app.services.jina import JinaEmbeddingService
from app.services.logo_detector import GroundingDINOService
from app.services.logo_llm_classifier import LogoLLMClassificationService
from app.services.logo_pipeline import LogoPipelineService
from app.services.net import NetworkService
from app.services.qdrant_store import QdrantVectorService
from app.services.storage import ObjectStorageService


@lru_cache
def get_object_storage_service() -> ObjectStorageService:
    return ObjectStorageService()


@lru_cache
def get_embedding_service() -> JinaEmbeddingService:
    return JinaEmbeddingService()


@lru_cache
def get_qdrant_service() -> QdrantVectorService:
    return QdrantVectorService()


@lru_cache
def get_network_service() -> NetworkService:
    return NetworkService()


@lru_cache
def get_logo_detector_service() -> GroundingDINOService:
    return GroundingDINOService()


@lru_cache
def get_logo_pipeline_service() -> LogoPipelineService:
    return LogoPipelineService(
        detector=get_logo_detector_service(),
        embeddings=get_embedding_service(),
        vector_store=get_qdrant_service(),
    )


@lru_cache
def get_logo_llm_classification_service() -> LogoLLMClassificationService:
    return LogoLLMClassificationService(
        storage=get_object_storage_service(),
        model=settings.MODEL_IMAGE_DETAIL,
    )
