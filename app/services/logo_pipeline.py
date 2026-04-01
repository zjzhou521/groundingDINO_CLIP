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
    winning_reference_image_id: str | None
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
            "winning_reference_image_id": self.winning_reference_image_id,
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

    def detect(self, image: Image.Image, top_k: int | None = None) -> list[DetectedLogoBox]:
        """Detect top K logos in the image."""
        return self.detector.detect(image, top_k=top_k)

    def classify(self, user_id: str, image: Image.Image) -> ClassificationOutcome:
        detections = self.detect(image)
        used_full_image_fallback = False
        target_images: list[Image.Image] = []
        target_detections: list[DetectedLogoBox | None] = []

        if detections:
            for detection in detections:
                crop_box = clamp_box(
                    (detection.x_min, detection.y_min, detection.x_max, detection.y_max),
                    image.width,
                    image.height,
                )
                target_images.append(image.crop(crop_box))
                target_detections.append(detection)
        elif settings.CLASSIFICATION_FALLBACK_TO_FULL_IMAGE:
            target_images.append(image)
            target_detections.append(None)
            used_full_image_fallback = True
        else:
            return ClassificationOutcome(
                detection=None,
                predicted_logo_id=None,
                predicted_logo_name=None,
                score=None,
                margin=None,
                winning_reference_image_id=None,
                matched=False,
                used_full_image_fallback=False,
                candidates=[],
            )

        data_urls: list[str] = []
        for target_image in target_images:
            image_bytes, content_type = image_to_png_bytes(target_image)
            data_urls.append(image_bytes_to_data_url(image_bytes, content_type))

        vectors = self.embeddings.embed_images(data_urls)
        search_results: list[tuple[DetectedLogoBox | None, list]] = []
        for detection, vector in zip(target_detections, vectors, strict=True):
            neighbors = self.vector_store.search_reference_embeddings(
                user_id=user_id,
                vector=vector,
                limit=settings.CLASSIFICATION_TOP_K,
            )
            search_results.append((detection, neighbors))

        candidates, best_detections, best_reference_image_ids = self._aggregate_candidates(
            search_results
        )
        if not candidates:
            return ClassificationOutcome(
                detection=detections[0] if detections else None,
                predicted_logo_id=None,
                predicted_logo_name=None,
                score=None,
                margin=None,
                winning_reference_image_id=None,
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
            detection=best_detections.get(top_one.logo_id),
            predicted_logo_id=top_one.logo_id if matched else None,
            predicted_logo_name=top_one.logo_name if matched else None,
            score=top_one.score,
            margin=margin,
            winning_reference_image_id=best_reference_image_ids.get(top_one.logo_id),
            matched=matched,
            used_full_image_fallback=used_full_image_fallback,
            candidates=candidates,
        )

    def _aggregate_candidates(
        self,
        search_results: list[tuple[DetectedLogoBox | None, list]],
    ) -> tuple[
        list[CandidateMatch],
        dict[str, DetectedLogoBox | None],
        dict[str, str | None],
    ]:
        grouped: dict[str, dict] = defaultdict(
            lambda: {
                "logo_name": None,
                "best_score": float("-inf"),
                "best_detection": None,
                "best_reference_image_id": None,
                "reference_image_ids": set(),
            }
        )

        for detection, neighbors in search_results:
            crop_grouped: dict[str, dict] = defaultdict(
                lambda: {
                    "logo_name": None,
                    "scores": [],
                    "reference_image_ids": [],
                    "best_reference_image_id": None,
                    "best_reference_score": float("-inf"),
                }
            )

            for neighbor in neighbors:
                payload = neighbor.payload or {}
                logo_id = payload.get("logo_id")
                if not logo_id:
                    continue

                crop_group = crop_grouped[logo_id]
                crop_group["logo_name"] = payload.get("logo_name")
                neighbor_score = float(neighbor.score)
                crop_group["scores"].append(neighbor_score)
                reference_image_id = payload.get("reference_image_id")
                if reference_image_id is not None:
                    crop_group["reference_image_ids"].append(reference_image_id)
                    if neighbor_score > crop_group["best_reference_score"]:
                        crop_group["best_reference_score"] = neighbor_score
                        crop_group["best_reference_image_id"] = reference_image_id

            for logo_id, values in crop_grouped.items():
                scores = sorted(values["scores"], reverse=True)[:3]
                if not scores:
                    continue

                crop_score = sum(scores) / len(scores)
                group = grouped[logo_id]
                group["logo_name"] = values["logo_name"] or group["logo_name"] or logo_id
                group["reference_image_ids"].update(values["reference_image_ids"])
                if crop_score > group["best_score"]:
                    group["best_score"] = crop_score
                    group["best_detection"] = detection
                    group["best_reference_image_id"] = values["best_reference_image_id"]

        candidates: list[CandidateMatch] = []
        best_detections: dict[str, DetectedLogoBox | None] = {}
        best_reference_image_ids: dict[str, str | None] = {}
        for logo_id, values in grouped.items():
            candidates.append(
                CandidateMatch(
                    logo_id=logo_id,
                    logo_name=values["logo_name"] or logo_id,
                    score=values["best_score"],
                    reference_image_ids=sorted(values["reference_image_ids"]),
                )
            )
            best_detections[logo_id] = values["best_detection"]
            best_reference_image_ids[logo_id] = values["best_reference_image_id"]

        candidates = sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.logo_name, candidate.logo_id),
        )[: settings.CLASSIFICATION_TOP_K]
        best_detections = {
            candidate.logo_id: best_detections.get(candidate.logo_id) for candidate in candidates
        }
        best_reference_image_ids = {
            candidate.logo_id: best_reference_image_ids.get(candidate.logo_id)
            for candidate in candidates
        }
        return candidates, best_detections, best_reference_image_ids
