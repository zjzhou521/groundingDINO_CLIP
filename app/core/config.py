from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "logo-detection-service"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str

    JINA_API_KEY: SecretStr
    JINA_EMBEDDING_MODEL: str = "jina-clip-v2"
    JINA_EMBEDDINGS_URL: str = "https://api.jina.ai/v1/embeddings"
    JINA_BATCH_SIZE: int = 8
    JINA_TIMEOUT_SECONDS: float = 30.0
    JINA_USE_PROXY: bool = False

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_COLLECTION_NAME: str = "logo_reference_vectors"
    QDRANT_USE_PROXY: bool = False

    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY_ID: str = "minioadmin"
    S3_SECRET_ACCESS_KEY: SecretStr = SecretStr("minioadmin")
    S3_BUCKET_NAME: str = "logo-images"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False
    S3_USE_PROXY: bool = False

    DOWNLOAD_USE_PROXY: bool = False

    GROUNDING_DINO_MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"
    GROUNDING_DINO_DEVICE: str = "cpu"
    DETECTION_PROMPT: str = "logo . brand logo . emblem . trademark . label"
    DETECTION_BOX_THRESHOLD: float = 0.25
    DETECTION_TEXT_THRESHOLD: float = 0.20
    DETECTION_TOP_K: int = 5

    CLASSIFICATION_TOP_K: int = 8
    CLASSIFICATION_MATCH_THRESHOLD: float = 0.45
    CLASSIFICATION_MARGIN_THRESHOLD: float = 0.03
    CLASSIFICATION_FALLBACK_TO_FULL_IMAGE: bool = True

    LLM_BASE_URL: str
    LLM_API_KEY: SecretStr
    MODEL_IMAGE_DETAIL: str


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
