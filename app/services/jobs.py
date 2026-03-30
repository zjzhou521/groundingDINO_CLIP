import uuid

from sqlalchemy.orm import Session

from app.models import Job


def create_job(
    db: Session,
    *,
    user_id: str,
    job_type: str,
    params: dict,
    logo_id: uuid.UUID | None = None,
    product_id: uuid.UUID | None = None,
) -> Job:
    job = Job(
        user_id=user_id,
        job_type=job_type,
        status="pending",
        params=params,
        logo_id=logo_id,
        product_id=product_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def mark_job_running(db: Session, job: Job) -> Job:
    job.status = "running"
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def mark_job_succeeded(db: Session, job: Job, result: dict) -> Job:
    job.status = "succeeded"
    job.result = result
    job.error_message = None
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def mark_job_failed(db: Session, job: Job, error_message: str) -> Job:
    db.rollback()
    job.status = "failed"
    job.error_message = error_message[:4000]
    db.add(job)
    db.commit()
    db.refresh(job)
    return job
