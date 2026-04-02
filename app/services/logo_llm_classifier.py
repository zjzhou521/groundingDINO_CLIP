import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Logo
from app.services.llm import llm
from app.services.storage import ObjectStorageService
from app.utils.images import image_bytes_to_data_url

logger = logging.getLogger(__name__)

LOGO_LLM_SYSTEM_PROMPT = (
    "You are a logo classification assistant. "
    "You will receive several user messages, where each earlier user message describes one "
    "known logo class and includes up to five reference images for that class. "
    "The final user message contains the image to classify. "
    "Classify the final image into one of the provided logo classes only. "
    "Respond with only a JSON object using this exact schema: "
    '{"logo_name":"<one provided logo name or unknown>","confidence":<float_between_0_and_1>}. '
    "The confidence must be a float from 0 to 1. "
    "Do not include markdown, code fences, or any explanation. "
    "Calibrate confidence honestly after comparing the best class against the closest alternative. "
    "Use this rubric: 0.98-1.00 only for an almost certain match with no plausible alternative; "
    "0.90-0.97 for a clear match with only minor variation; "
    "0.80-0.89 for a likely match with some ambiguity; "
    "0.60-0.79 for a tentative match; "
    "below 0.60 for a weak or highly ambiguous match. "
    "If none of the classes match, or you are genuinely unsure, use unknown. "
    "Do not guess, do not randomly choose a provided class, do not default to 0.95, "
    "and do not reuse the same confidence across different images unless the "
    "evidence strength is truly similar."
)


def extract_logo_llm_token_cost(chat_completion: ChatCompletion) -> Any | None:
    if chat_completion.usage is not None:
        return chat_completion.usage.model_dump(exclude_none=True)

    for choice in chat_completion.choices:
        choice_payload = choice.model_dump(exclude_none=True)
        for key in ("usage", "token_cost", "token_usage", "usage_metadata"):
            value = choice_payload.get(key)
            if value is not None:
                return value

        message_payload = choice_payload.get("message")
        if not isinstance(message_payload, dict):
            continue

        for key in ("usage", "token_cost", "token_usage", "usage_metadata"):
            value = message_payload.get(key)
            if value is not None:
                return value

    return None


@dataclass
class LLMClassReference:
    logo_id: str
    logo_name: str
    reference_image_ids: list[str]
    reference_image_data_urls: list[str]


@dataclass
class LLMClassificationOutcome:
    predicted_logo_id: str | None
    predicted_logo_name: str | None
    confidence: float | None
    raw_response: str
    token_cost: Any | None
    reference_class_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_logo_id": self.predicted_logo_id,
            "predicted_logo_name": self.predicted_logo_name,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "token_cost": self.token_cost,
            "reference_class_count": self.reference_class_count,
        }


class LogoLLMClassificationService:
    def __init__(self, *, storage: ObjectStorageService, model: str) -> None:
        self.storage = storage
        self.model = model

    def classify(
        self,
        *,
        db: Session,
        user_id: str,
        image_bytes: bytes,
        content_type: str,
    ) -> LLMClassificationOutcome:
        reference_classes = self._load_reference_classes(db=db, user_id=user_id)
        if not reference_classes:
            raise ValueError("No readable logo reference images found for this user")

        messages, class_lookup = self._build_messages(
            reference_classes=reference_classes,
            image_bytes=image_bytes,
            content_type=content_type,
        )
        raw_response, token_cost = asyncio.run(self._classify_async(messages))
        raw_response = raw_response.strip()
        parsed_response = self._parse_structured_response(raw_response)
        response_logo_name = self._extract_logo_name(parsed_response, raw_response)
        confidence = self._extract_confidence(parsed_response)
        matched_class = self._match_logo_name(response_logo_name, class_lookup)

        if matched_class is None:
            logger.info(
                "LLM logo classification did not map to a known class: user_id=%s response=%r",
                user_id,
                raw_response,
            )
            return LLMClassificationOutcome(
                predicted_logo_id=None,
                predicted_logo_name=None,
                confidence=confidence,
                raw_response=raw_response,
                token_cost=token_cost,
                reference_class_count=len(reference_classes),
            )

        return LLMClassificationOutcome(
            predicted_logo_id=matched_class.logo_id,
            predicted_logo_name=matched_class.logo_name,
            confidence=confidence,
            raw_response=raw_response,
            token_cost=token_cost,
            reference_class_count=len(reference_classes),
        )

    async def _classify_async(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> tuple[str, Any | None]:
        openai_client = llm()
        chat_completion = await openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        response = chat_completion.choices[0].message.content
        return (
            self._coerce_response_text(response),
            extract_logo_llm_token_cost(chat_completion),
        )

    def _load_reference_classes(
        self,
        *,
        db: Session,
        user_id: str,
    ) -> list[LLMClassReference]:
        logos = db.scalars(
            select(Logo)
            .where(Logo.user_id == user_id)
            .options(selectinload(Logo.reference_images))
            .order_by(Logo.name)
        ).all()

        reference_classes: list[LLMClassReference] = []
        for logo in logos:
            reference_images = sorted(
                logo.reference_images,
                key=lambda reference: str(reference.id),
            )
            data_urls: list[str] = []
            reference_image_ids: list[str] = []

            for reference in reference_images[:5]:
                try:
                    image_bytes = self.storage.download_bytes(reference.storage_key)
                except Exception:
                    logger.exception(
                        (
                            "Failed to load reference image from storage: "
                            "logo_id=%s reference_image_id=%s storage_key=%s"
                        ),
                        str(logo.id),
                        str(reference.id),
                        reference.storage_key,
                    )
                    continue

                data_urls.append(
                    image_bytes_to_data_url(image_bytes, reference.content_type or "image/png")
                )
                reference_image_ids.append(str(reference.id))

            if not data_urls:
                continue

            reference_classes.append(
                LLMClassReference(
                    logo_id=str(logo.id),
                    logo_name=logo.name,
                    reference_image_ids=reference_image_ids,
                    reference_image_data_urls=data_urls,
                )
            )

        return reference_classes

    def _build_messages(
        self,
        *,
        reference_classes: list[LLMClassReference],
        image_bytes: bytes,
        content_type: str,
    ) -> tuple[list[ChatCompletionMessageParam], dict[str, list[LLMClassReference]]]:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": LOGO_LLM_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        class_lookup: dict[str, list[LLMClassReference]] = {}
        logo_names: list[str] = []
        for reference_class in reference_classes:
            normalized_name = self._normalize_logo_name(reference_class.logo_name)
            class_lookup.setdefault(normalized_name, []).append(reference_class)
            logo_names.append(reference_class.logo_name)

            content = [
                {
                    "type": "text",
                    "text": (
                        f"Here are {len(reference_class.reference_image_data_urls)} "
                        "reference images "
                        f"that belong to the {reference_class.logo_name} brand logo. "
                        f"Remember this class name exactly as: {reference_class.logo_name}"
                    ),
                }
            ]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "low"},
                }
                for data_url in reference_class.reference_image_data_urls
            )
            messages.append({"role": "user", "content": content})

        query_image_url = image_bytes_to_data_url(image_bytes, content_type)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Classify this image into exactly one of the following logo names: "
                            + ", ".join(logo_names)
                            + ". Reply only with JSON like "
                            '{"logo_name":"chosen_logo_or_unknown",'
                            '"confidence":<calibrated_float>}. '
                            "The confidence must be a float from 0 to 1 and must vary based on "
                            "how clearly the image matches the best class versus the closest "
                            "alternative. Use unknown only when none of the classes fit or you "
                            "are genuinely unsure. Never default to 0.95, and do not reuse the "
                            "same confidence across different images unless the evidence strength "
                            "is actually similar."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": query_image_url, "detail": "low"},
                    },
                ],
            }
        )
        return messages, class_lookup

    def _match_logo_name(
        self,
        response_logo_name: str | None,
        class_lookup: dict[str, list[LLMClassReference]],
    ) -> LLMClassReference | None:
        if response_logo_name is None:
            return None

        cleaned = response_logo_name.strip().strip("`").strip().strip('"').strip("'")
        if not cleaned:
            return None
        if cleaned.casefold() == "unknown":
            return None

        for references in class_lookup.values():
            for reference in references:
                if cleaned == reference.logo_name:
                    return reference
                if cleaned.casefold() == reference.logo_name.casefold():
                    return reference
                if reference.logo_name.casefold() in cleaned.casefold():
                    return reference

        normalized = self._normalize_logo_name(cleaned)
        matches = class_lookup.get(normalized, [])
        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            logger.warning(
                "Ambiguous normalized LLM logo response: response=%r matched_logos=%s",
                response_logo_name,
                [match.logo_name for match in matches],
            )
        return None

    @staticmethod
    def _parse_structured_response(raw_response: str) -> dict[str, Any] | None:
        text = raw_response.strip()
        if not text:
            return None

        candidates = [text]
        if text.startswith("```"):
            fence_lines = text.splitlines()
            if len(fence_lines) >= 3:
                candidates.append("\n".join(fence_lines[1:-1]).strip())

        object_start = text.find("{")
        object_end = text.rfind("}")
        if object_start != -1 and object_end > object_start:
            candidates.append(text[object_start : object_end + 1].strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    @staticmethod
    def _extract_logo_name(
        parsed_response: dict[str, Any] | None,
        raw_response: str,
    ) -> str | None:
        if parsed_response is not None:
            value = parsed_response.get("logo_name")
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized

        text = raw_response.strip()
        return text or None

    @staticmethod
    def _extract_confidence(parsed_response: dict[str, Any] | None) -> float | None:
        if parsed_response is None:
            return None

        value = parsed_response.get("confidence")
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return max(0.0, min(float(value), 1.0))
        if isinstance(value, str):
            try:
                return max(0.0, min(float(value.strip()), 1.0))
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_response_text(response: str | list | None) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            text_parts: list[str] = []
            for item in response:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                    continue

                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            return "\n".join(text_parts).strip()
        return ""

    @staticmethod
    def _normalize_logo_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())
