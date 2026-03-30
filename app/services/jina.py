import logging
import asyncio
from collections.abc import Iterable, Sequence

import aiohttp
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class RetryableJinaError(RuntimeError):
    pass


def chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class JinaEmbeddingService:
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.JINA_API_KEY.get_secret_value()}",
        }

    def close(self) -> None:
        # No persistent HTTP client is held; kept for backwards-compatible interface.
        return

    def embed_images(self, data_urls: Sequence[str], task: str | None = None) -> list[list[float]]:
        if not data_urls:
            return []

        # Routes are sync functions (run in a threadpool), so we bridge into async here.
        return asyncio.run(self._embed_images_async(list(data_urls), task=task))

    async def _embed_images_async(self, data_urls: Sequence[str], task: str | None = None) -> list[list[float]]:
        timeout = aiohttp.ClientTimeout(total=settings.JINA_TIMEOUT_SECONDS)
        vectors: list[list[float]] = []

        # trust_env=True allows aiohttp to respect HTTP_PROXY/HTTPS_PROXY
        # Set JINA_USE_PROXY=false to disable proxy for Jina API calls
        async with aiohttp.ClientSession(
            timeout=timeout,
            trust_env=settings.JINA_USE_PROXY,
        ) as session:
            total = len(data_urls)
            for idx, batch in enumerate(chunks(data_urls, settings.JINA_BATCH_SIZE), start=1):
                logger.info(
                    "Embedding batch %d started: batch_size=%d total_images=%d timeout_seconds=%s",
                    idx,
                    len(batch),
                    total,
                    settings.JINA_TIMEOUT_SECONDS,
                )
                vectors.extend(await self._embed_batch_async(session, batch, task=task))
                logger.info(
                    "Embedding batch %d finished: cumulative_vectors=%d total_images=%d",
                    idx,
                    len(vectors),
                    total,
                )
        return vectors

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (
                RetryableJinaError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _post_embeddings_async(self, session: aiohttp.ClientSession, payload: dict) -> dict:
        input_count = len(payload.get("input", []))
        model = payload.get("model", "unknown")
        logger.debug("Sending embedding request to Jina: model=%s, items=%d", model, input_count)

        try:
            async with session.post(
                settings.JINA_EMBEDDINGS_URL,
                json=payload,
                headers=self.headers,
            ) as response:
                if response.status in self.RETRYABLE_STATUS_CODES:
                    logger.warning("Jina API returned retryable status %s", response.status)
                    raise RetryableJinaError(f"Jina API returned retryable status {response.status}")

                if response.status >= 400:
                    error_body = await response.text()
                    logger.error(
                        "Jina API error: status=%s, model=%s, items=%d, response=%s",
                        response.status,
                        model,
                        input_count,
                        error_body[:1000] if error_body else "<empty>",
                    )

                response.raise_for_status()
                body = await response.json()
        except asyncio.TimeoutError:
            logger.exception(
                "Jina request timed out: model=%s items=%d timeout_seconds=%s",
                model,
                input_count,
                settings.JINA_TIMEOUT_SECONDS,
            )
            raise
        except aiohttp.ClientError:
            logger.exception(
                "Jina client error: model=%s items=%d endpoint=%s",
                model,
                input_count,
                settings.JINA_EMBEDDINGS_URL,
            )
            raise
        logger.debug("Jina API request successful: items=%d", input_count)
        return body

    async def _embed_batch_async(
        self,
        session: aiohttp.ClientSession,
        data_urls: Sequence[str],
        task: str | None = None,
    ) -> list[list[float]]:
        # Jina API expects base64 data URLs as strings, or ImageDoc objects like {"image": "data:..."}.
        # Using the ImageDoc format for explicit image input typing.
        payload = {
            "model": settings.JINA_EMBEDDING_MODEL,
            "input": [{"image": item} for item in data_urls],
            "normalized": True,
            "embedding_type": "float",
        }
        if task is not None:
            payload["task"] = task

        response_payload = await self._post_embeddings_async(session, payload)
        rows = sorted(response_payload.get("data", []), key=lambda item: item["index"])
        return [row["embedding"] for row in rows]
