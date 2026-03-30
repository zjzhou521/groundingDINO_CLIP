import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import get_db
from app.dependencies import (
    get_embedding_service,
    get_logo_pipeline_service,
    get_network_service,
    get_object_storage_service,
    get_qdrant_service,
)
from app.models import Logo, LogoReferenceImage, Product
from app.schemas import (
    BoundingBox,
    ClassifyLogoResponse,
    DetectLogoResponse,
    HealthResponse,
    MatchCandidate,
    ProductImageRequest,
    ReferenceImageUploadRequest,
    ReferenceUploadResponse,
)
from app.services.jobs import create_job, mark_job_failed, mark_job_running, mark_job_succeeded
from app.services.net import RemoteImageDownloadError
from app.services.qdrant_store import ReferenceVectorRecord
from app.utils.images import image_bytes_to_data_url, image_to_png_bytes, load_image_from_bytes

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


def _download_and_store_image(
    *,
    image_url: str,
    storage_prefix: str,
    storage_owner: str,
    storage_entity_id: str,
):
    network = get_network_service()
    storage = get_object_storage_service()

    downloaded = network.download_image(image_url)
    image = load_image_from_bytes(downloaded.content)
    normalized_bytes, content_type = image_to_png_bytes(image)
    storage_key = f"{storage_prefix}/{storage_owner}/{storage_entity_id}/{uuid.uuid4()}.png"
    storage.upload_bytes(storage_key, normalized_bytes, content_type)
    storage_url = storage.build_object_url(storage_key)

    return downloaded, image, normalized_bytes, content_type, storage_key, storage_url


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/logos/reference-images", response_model=ReferenceUploadResponse)
def upload_reference_images(
    payload: ReferenceImageUploadRequest,
    db: DbSession,
) -> ReferenceUploadResponse:
    user_id = payload.user_id
    cleaned_logo_name = payload.logo_name.strip()
    if not cleaned_logo_name:
        raise HTTPException(status_code=400, detail="logo_name must not be empty")
    if not payload.image_urls:
        raise HTTPException(status_code=400, detail="At least one image_url is required")

    embeddings = get_embedding_service()
    vector_store = get_qdrant_service()

    logo = db.scalar(select(Logo).where(Logo.user_id == user_id, Logo.name == cleaned_logo_name))
    if logo is None:
        logo = Logo(user_id=user_id, name=cleaned_logo_name)
        db.add(logo)
        db.commit()
        db.refresh(logo)

    job = create_job(
        db,
        user_id=user_id,
        job_type="reference_upload",
        params={
            "logo_name": cleaned_logo_name,
            "image_urls": [str(image_url) for image_url in payload.image_urls],
        },
        logo_id=logo.id,
    )

    try:
        mark_job_running(db, job)
        data_urls: list[str] = []
        references: list[LogoReferenceImage] = []

        for image_url in payload.image_urls:
            downloaded, image, normalized_bytes, content_type, storage_key, storage_url = (
                _download_and_store_image(
                    image_url=str(image_url),
                    storage_prefix="reference",
                    storage_owner=user_id,
                    storage_entity_id=str(logo.id),
                )
            )

            reference = LogoReferenceImage(
                user_id=user_id,
                logo_id=logo.id,
                source_url=str(image_url),
                storage_key=storage_key,
                storage_url=storage_url,
                original_filename=downloaded.filename,
                content_type=content_type,
                width=image.width,
                height=image.height,
            )
            db.add(reference)
            db.commit()
            db.refresh(reference)

            references.append(reference)
            data_urls.append(image_bytes_to_data_url(normalized_bytes, content_type))

        if not references:
            raise HTTPException(
                status_code=400,
                detail="No readable reference images were uploaded",
            )
        logger.info(
            "Starting Jina embedding request: images=%d user_id=%s logo_id=%s job_id=%s timeout_seconds=%s",
            len(data_urls),
            user_id,
            str(logo.id),
            str(job.id),
            settings.JINA_TIMEOUT_SECONDS,
        )
        embedding_started = time.perf_counter()
        try:
            vectors = embeddings.embed_images(data_urls)
        except Exception:
            elapsed = time.perf_counter() - embedding_started
            logger.exception(
                "Jina embedding failed after %.2fs: images=%d user_id=%s logo_id=%s job_id=%s first_source_url=%s",
                elapsed,
                len(data_urls),
                user_id,
                str(logo.id),
                str(job.id),
                str(payload.image_urls[0]) if payload.image_urls else "<none>",
            )
            raise
        logger.info(
            "Jina embedding completed in %.2fs: vectors=%d user_id=%s logo_id=%s job_id=%s",
            time.perf_counter() - embedding_started,
            len(vectors),
            user_id,
            str(logo.id),
            str(job.id),
        )
        vector_records = [
            ReferenceVectorRecord(
                point_id=str(reference.id),
                user_id=user_id,
                logo_id=str(logo.id),
                logo_name=logo.name,
                reference_image_id=str(reference.id),
                vector=vector,
            )
            for reference, vector in zip(references, vectors, strict=True)
        ]
        vector_store.upsert_reference_embeddings(vector_records)

        for reference in references:
            reference.qdrant_point_id = str(reference.id)
            db.add(reference)
        db.commit()

        result = {
            "logo_id": str(logo.id),
            "logo_name": logo.name,
            "uploaded_images": len(references),
            "qdrant_points_upserted": len(vector_records),
        }
        mark_job_succeeded(db, job, result)

        return ReferenceUploadResponse(job_id=str(job.id), **result)
    except RemoteImageDownloadError as exc:
        logger.error("RemoteImageDownloadError: %s", exc)
        mark_job_failed(db, job, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Reference upload job failed: user_id=%s logo_name=%s logo_id=%s job_id=%s error=%s",
            user_id,
            cleaned_logo_name,
            str(logo.id),
            str(job.id),
            exc,
        )
        mark_job_failed(db, job, str(exc))
        raise


@router.post("/products/detect-logo", response_model=DetectLogoResponse)
def detect_logo(
    payload: ProductImageRequest,
    db: DbSession,
) -> DetectLogoResponse:
    pipeline = get_logo_pipeline_service()
    user_id = payload.user_id
    job = create_job(
        db,
        user_id=user_id,
        job_type="detect_logo",
        params={"image_url": str(payload.image_url)},
    )

    try:
        mark_job_running(db, job)
        product_id = uuid.uuid4()
        downloaded, image, _, content_type, storage_key, storage_url = _download_and_store_image(
            image_url=str(payload.image_url),
            storage_prefix="products",
            storage_owner=user_id,
            storage_entity_id=str(product_id),
        )

        product = Product(
            id=product_id,
            user_id=user_id,
            source_url=str(payload.image_url),
            storage_key=storage_key,
            storage_url=storage_url,
            original_filename=downloaded.filename,
            content_type=content_type,
            width=image.width,
            height=image.height,
        )
        db.add(product)
        db.commit()
        db.refresh(product)

        job.product_id = product.id
        db.add(job)
        db.commit()
        db.refresh(job)

        detections = pipeline.detect(image, top_k=5)
        result = {
            "product_id": str(product.id),
            "detections": [d.as_dict() for d in detections],
            "found": len(detections) > 0,
        }
        mark_job_succeeded(db, job, result)

        return DetectLogoResponse(
            job_id=str(job.id),
            product_id=str(product.id),
            detections=[BoundingBox(**d.as_dict()) for d in detections],
            found=len(detections) > 0,
        )
    except RemoteImageDownloadError as exc:
        logger.error("RemoteImageDownloadError: %s", exc)
        mark_job_failed(db, job, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        mark_job_failed(db, job, str(exc))
        raise


@router.post("/products/detect-logo-file", response_model=DetectLogoResponse)
def detect_logo_file(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DetectLogoResponse:
    """Detect logo by uploading a file directly."""
    pipeline = get_logo_pipeline_service()
    storage = get_object_storage_service()

    job = create_job(
        db,
        user_id=user_id,
        job_type="detect_logo_file",
        params={"filename": file.filename},
    )

    try:
        mark_job_running(db, job)

        # Read uploaded file
        content = file.file.read()
        image = load_image_from_bytes(content)
        normalized_bytes, content_type = image_to_png_bytes(image)

        product_id = uuid.uuid4()
        storage_key = f"products/{user_id}/{product_id}/{uuid.uuid4()}.png"
        storage.upload_bytes(storage_key, normalized_bytes, content_type)
        storage_url = storage.build_object_url(storage_key)

        product = Product(
            id=product_id,
            user_id=user_id,
            source_url=f"upload://{file.filename}",
            storage_key=storage_key,
            storage_url=storage_url,
            original_filename=file.filename or "uploaded-image",
            content_type=content_type,
            width=image.width,
            height=image.height,
        )
        db.add(product)
        db.commit()
        db.refresh(product)

        job.product_id = product.id
        db.add(job)
        db.commit()
        db.refresh(job)

        detections = pipeline.detect(image, top_k=5)
        result = {
            "product_id": str(product.id),
            "detections": [d.as_dict() for d in detections],
            "found": len(detections) > 0,
        }
        mark_job_succeeded(db, job, result)

        return DetectLogoResponse(
            job_id=str(job.id),
            product_id=str(product.id),
            detections=[BoundingBox(**d.as_dict()) for d in detections],
            found=len(detections) > 0,
        )
    except Exception as exc:
        logger.exception("File upload detection failed: %s", exc)
        mark_job_failed(db, job, str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/products/classify-logo", response_model=ClassifyLogoResponse)
def classify_logo(
    payload: ProductImageRequest,
    db: DbSession,
) -> ClassifyLogoResponse:
    pipeline = get_logo_pipeline_service()
    user_id = payload.user_id
    job = create_job(
        db,
        user_id=user_id,
        job_type="classify_logo",
        params={"image_url": str(payload.image_url)},
    )

    try:
        mark_job_running(db, job)
        product_id = uuid.uuid4()
        downloaded, image, _, content_type, storage_key, storage_url = _download_and_store_image(
            image_url=str(payload.image_url),
            storage_prefix="products",
            storage_owner=user_id,
            storage_entity_id=str(product_id),
        )

        product = Product(
            id=product_id,
            user_id=user_id,
            source_url=str(payload.image_url),
            storage_key=storage_key,
            storage_url=storage_url,
            original_filename=downloaded.filename,
            content_type=content_type,
            width=image.width,
            height=image.height,
        )
        db.add(product)
        db.commit()
        db.refresh(product)

        job.product_id = product.id
        db.add(job)
        db.commit()
        db.refresh(job)

        outcome = pipeline.classify(user_id=user_id, image=image)
        result = {"product_id": str(product.id), **outcome.as_dict()}
        mark_job_succeeded(db, job, result)

        return ClassifyLogoResponse(
            job_id=str(job.id),
            product_id=str(product.id),
            detection=BoundingBox(**outcome.detection.as_dict()) if outcome.detection else None,
            predicted_logo_id=outcome.predicted_logo_id,
            predicted_logo_name=outcome.predicted_logo_name,
            score=outcome.score,
            margin=outcome.margin,
            matched=outcome.matched,
            used_full_image_fallback=outcome.used_full_image_fallback,
            candidates=[MatchCandidate(**candidate.as_dict()) for candidate in outcome.candidates],
        )
    except RemoteImageDownloadError as exc:
        logger.error("RemoteImageDownloadError: %s", exc)
        mark_job_failed(db, job, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        mark_job_failed(db, job, str(exc))
        raise
