from dataclasses import dataclass

import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from app.core.config import settings


@dataclass
class DetectedLogoBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    label: str | None

    def as_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "score": self.score,
            "label": self.label,
        }


class GroundingDINOService:
    def __init__(self) -> None:
        self.device = settings.GROUNDING_DINO_DEVICE
        self.model_id = settings.GROUNDING_DINO_MODEL_ID
        self.labels = [
            label.strip() for label in settings.DETECTION_PROMPT.split(".") if label.strip()
        ]
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        # Use AutoProcessor with trust_remote_code for GroundingDINO
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id,
            trust_remote_code=True
        ).to(self.device)
        self._model.eval()

    def detect(self, image: Image.Image, top_k: int = 5) -> list[DetectedLogoBox]:
        """Detect top K logos in the image."""
        self._load()

        # Build the text prompt for grounding (single string with labels)
        text = " . ".join(self.labels)

        inputs = self._processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=settings.DETECTION_BOX_THRESHOLD,
            text_threshold=settings.DETECTION_TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )

        result = results[0]
        num_boxes = len(result["boxes"])
        if num_boxes == 0:
            return []

        scores = result["scores"]
        labels = result.get("labels") or []

        # Get indices sorted by score (descending)
        sorted_indices = sorted(
            range(num_boxes),
            key=lambda index: float(scores[index]),
            reverse=True
        )[:top_k]

        detections: list[DetectedLogoBox] = []
        for index in sorted_indices:
            box = result["boxes"][index].tolist()
            label = labels[index] if index < len(labels) else None
            detections.append(
                DetectedLogoBox(
                    x_min=float(box[0]),
                    y_min=float(box[1]),
                    x_max=float(box[2]),
                    y_max=float(box[3]),
                    score=float(scores[index]),
                    label=label,
                )
            )

        return detections
