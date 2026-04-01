import os
from contextlib import contextmanager
from datetime import timedelta

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings


@contextmanager
def _clear_proxy_env():
    """Temporarily clear proxy-related environment variables."""
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    saved = {var: os.environ.pop(var, None) for var in proxy_vars}
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is not None:
                os.environ[var] = value


class ObjectStorageService:
    def __init__(self) -> None:
        self.bucket_name = settings.S3_BUCKET_NAME

        # Configure proxy settings for S3/MinIO
        # When S3_USE_PROXY is False, clear proxy env vars before creating client
        # When True, boto3 will automatically use HTTP_PROXY/HTTPS_PROXY from environment
        if settings.S3_USE_PROXY:
            self.client = self._create_s3_client()
        else:
            # Clear proxy env vars to ensure direct connection to local MinIO
            with _clear_proxy_env():
                self.client = self._create_s3_client()

    def _create_s3_client(self):
        """Create and return the boto3 S3 client."""
        return boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
            region_name=settings.S3_REGION,
            use_ssl=settings.S3_USE_SSL,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)

    def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def build_object_url(self, key: str) -> str:
        base_url = settings.S3_ENDPOINT_URL.rstrip("/")
        return f"{base_url}/{self.bucket_name}/{key.lstrip('/')}"

    def generate_presigned_get_url(
        self,
        key: str,
        *,
        expires_in: int = int(timedelta(days=7).total_seconds()),
    ) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": key,
            },
            ExpiresIn=expires_in,
        )
