from pydantic import BaseModel, HttpUrl


class ReferenceImageUploadRequest(BaseModel):
    user_id: str
    logo_name: str
    image_urls: list[HttpUrl]


class ProductImageRequest(BaseModel):
    user_id: str
    image_url: HttpUrl


class HealthResponse(BaseModel):
    status: str


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    label: str | None = None


class MatchCandidate(BaseModel):
    logo_id: str
    logo_name: str
    score: float
    reference_image_ids: list[str]


class ReferenceUploadResponse(BaseModel):
    job_id: str
    logo_id: str
    logo_name: str
    uploaded_images: int
    qdrant_points_upserted: int


class DetectLogoResponse(BaseModel):
    job_id: str
    product_id: str
    detections: list[BoundingBox]
    found: bool


class ClassifyLogoResponse(BaseModel):
    job_id: str
    product_id: str
    detection: BoundingBox | None
    predicted_logo_id: str | None
    predicted_logo_name: str | None
    score: float | None
    margin: float | None
    matched: bool
    used_full_image_fallback: bool
    candidates: list[MatchCandidate]
