"""Network utilities for downloading images and other HTTP operations."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DownloadedImage:
    source_url: str
    content: bytes
    content_type: str
    filename: str


class RemoteImageDownloadError(RuntimeError):
    """Raised when image download fails."""
    pass


class NetworkService:
    """Service for HTTP operations with configurable proxy settings."""

    def __init__(self) -> None:
        pass

    def close(self) -> None:
        """Cleanup resources."""
        pass

    def download_image(self, image_url: str) -> DownloadedImage:
        """Download image from URL (synchronous wrapper around async implementation)."""
        return asyncio.run(self._download_image_async(image_url))

    async def _download_image_async(self, image_url: str) -> DownloadedImage:
        """Download image from URL using aiohttp."""
        headers = {"User-Agent": "logo-detection-service/0.1"}

        # Configure proxy: trust_env=True respects HTTP_PROXY env vars
        # trust_env=False ignores them for direct connections
        trust_env = settings.DOWNLOAD_USE_PROXY

        timeout = aiohttp.ClientTimeout(total=30.0)

        try:
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=trust_env
            ) as session:
                async with session.get(image_url, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                    content_type = response.headers.get(
                        "Content-Type", "application/octet-stream"
                    ).split(";")[0]
        except aiohttp.ClientResponseError as exc:
            logger.error(
                "HTTP error %s downloading image from %s: %s", exc.status, image_url, exc
            )
            raise RemoteImageDownloadError(
                f"Failed to download image: {image_url} (HTTP {exc.status})"
            ) from exc
        except aiohttp.ClientError as exc:
            logger.error("Client error downloading image from %s: %s", image_url, exc)
            raise RemoteImageDownloadError(
                f"Failed to download image: {image_url} ({exc})"
            ) from exc
        except asyncio.TimeoutError:
            logger.error("Timeout downloading image from %s", image_url)
            raise RemoteImageDownloadError(
                f"Failed to download image: {image_url} (timeout)"
            )
        except Exception as exc:
            logger.error("Unexpected error downloading image from %s: %s", image_url, exc)
            raise RemoteImageDownloadError(f"Failed to download image: {image_url}") from exc

        parsed_url = urlparse(image_url)
        filename = Path(unquote(parsed_url.path)).name or "downloaded-image"

        return DownloadedImage(
            source_url=image_url,
            content=content,
            content_type=content_type,
            filename=filename,
        )
