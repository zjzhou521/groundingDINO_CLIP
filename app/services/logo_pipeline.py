from collections import defaultdict
from dataclasses import dataclass

from PIL import Image

from app.core.config import settings
from app.services.jina import JinaEmbeddingService
from app.services.logo_detector import DetectedLogoBox, GroundingDINOService
from app.services.qdrant_store import QdrantVectorService
from app.utils.images import clamp_box, image_bytes_to_data_url, image_to_png_bytes


@dataclass
class CandidateMatch:
    logo_id: str
    logo_name: str
    score: float
    reference_image_ids: list[str]

    def as_dict(self) -> dict:
        return {
            "logo_id": self.logo_id,
            "logo_name": self.logo_name,
            "score": self.score,
            "reference_image_ids": self.reference_image_ids,
        }


@dataclass
class ClassificationOutcome:
    detection: DetectedLogoBox | None
    predicted_logo_id: str | None
    predicted_logo_name: str | None
    score: float | None
    margin: float | None
    matched: bool
    used_full_image_fallback: bool
    candidates: list[CandidateMatch]

    def as_dict(self) -> dict:
        return {
            "detection": self.detection.as_dict() if self.detection else None,
            "predicted_logo_id": self.predicted_logo_id,
            "predicted_logo_name": self.predicted_logo_name,
            "score": self.score,
            "margin": self.margin,
            "matched": self.matched,
            "used_full_image_fallback": self.used_full_image_fallback,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


class LogoPipelineService:
    def __init__(
        self,
        *,
        detector: GroundingDINOService,
        embeddings: JinaEmbeddingService,
        vector_store: QdrantVectorService,
    ) -> None:
        self.detector = detector
        self.embeddings = embeddings
        self.vector_store = vector_store

    def detect(self, image: Image.Image) -> DetectedLogoBox | None:
        return self.detector.detect_primary_logo(image)

    def classify(self, user_id: str, image: Image.Image) -> ClassificationOutcome:
        detection = self.detect(image)
        used_full_image_fallback = False

        if detection is not None:
            crop_box = clamp_box(
                (detection.x_min, detection.y_min, detection.x_max, detection.y_max),
                image.width,
                image.height,
            )
            target_image = image.crop(crop_box)
        elif settings.CLASSIFICATION_FALLBACK_TO_FULL_IMAGE:
            target_image = image
            used_full_image_fallback = True
        else:
            return ClassificationOutcome(
                detection=None,
                predicted_logo_id=None,
                predicted_logo_name=None,
                score=None,
                margin=None,
                matched=False,
                used_full_image_fallback=False,
                candidates=[],
            )

        image_bytes, content_type = image_to_png_bytes(target_image)
        data_url = image_bytes_to_data_url(image_bytes, content_type)
        vector = self.embeddings.embed_images([data_url])[0]
        neighbors = self.vector_store.search_reference_embeddings(
            user_id=user_id,
            vector=vector,
            limit=settings.CLASSIFICATION_TOP_K,
        )

        candidates = self._aggregate_candidates(neighbors)
        if not candidates:
            return ClassificationOutcome(
                detection=detection,
                predicted_logo_id=None,
                predicted_logo_name=None,
                score=None,
                margin=None,
                matched=False,
                used_full_image_fallback=used_full_image_fallback,
                candidates=[],
            )

        top_one = candidates[0]
        top_two = candidates[1] if len(candidates) > 1 else None
        margin = top_one.score - top_two.score if top_two else top_one.score
        matched = (
            top_one.score >= settings.CLASSIFICATION_MATCH_THRESHOLD
            and margin >= settings.CLASSIFICATION_MARGIN_THRESHOLD
        )

        return ClassificationOutcome(
            detection=detection,
            predicted_logo_id=top_one.logo_id if matched else None,
            predicted_logo_name=top_one.logo_name if matched else None,
            score=top_one.score,
            margin=margin,
            matched=matched,
            used_full_image_fallback=used_full_image_fallback,
            candidates=candidates,
        )

    def _aggregate_candidates(self, neighbors: list) -> list[CandidateMatch]:
        grouped: dict[str, dict] = defaultdict(
            lambda: {"logo_name": None, "scores": [], "reference_image_ids": []}
        )

        for neighbor in neighbors:
            payload = neighbor.payload or {}
            logo_id = payload.get("logo_id")
            if not logo_id:
                continue

            group = grouped[logo_id]
            group["logo_name"] = payload.get("logo_name")
            group["scores"].append(float(neighbor.score))
            reference_image_id = payload.get("reference_image_id")
            if reference_image_id is not None:
                group["reference_image_ids"].append(reference_image_id)

        candidates: list[CandidateMatch] = []
        for logo_id, values in grouped.items():
            scores = sorted(values["scores"], reverse=True)[:3]
            aggregate_score = sum(scores) / len(scores)
            candidates.append(
                CandidateMatch(
                    logo_id=logo_id,
                    logo_name=values["logo_name"] or logo_id,
                    score=aggregate_score,
                    reference_image_ids=values["reference_image_ids"],
                )
            )

        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
