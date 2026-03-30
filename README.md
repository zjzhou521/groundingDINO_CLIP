# Logo Detection Service

Python backend for logo enrollment, logo detection, and logo classification.

The service is built for this workflow:

1. A user uploads a small set of reference logos by passing public image URLs.
2. The backend downloads those images, stores them in S3-compatible object storage, and generates embeddings with Jina.
3. A user submits a product image URL.
4. GroundingDINO finds the most likely logo region.
5. The detected crop is embedded with Jina and matched against the enrolled logo vectors in Qdrant.
6. The API returns the logo position and the predicted logo name, or `unknown` if confidence is too low.

## Stack

- `FastAPI` for the HTTP API
- `Postgres` for relational data
- `Alembic` for schema migrations
- `Qdrant` for vector search
- `MinIO` or any S3-compatible storage for images
- `GroundingDINO` on CPU for logo detection
- `Jina` with `jina-clip-v2` for embeddings
- `Scalar` for API docs at `/docs`

## What The Service Stores

Postgres tables:

- `logo`
- `logo_reference_images`
- `products`
- `jobs`

Object storage:

- downloaded reference images
- downloaded product images

Qdrant:

- one vector per reference image
- payload includes `user_id`, `logo_id`, `logo_name`, and `reference_image_id`

## Project Structure

```text
app/
  api/          FastAPI routes
  core/         app config
  db/           SQLAlchemy session setup
  models/       ORM models
  schemas/      request and response schemas
  services/     Jina, Qdrant, S3, detection, pipeline services
  utils/        image helpers
alembic/        DB migrations
docker-compose.yml
Makefile
pyproject.toml
```

## Requirements

- Python `>= 3.11`
- [`uv`](https://docs.astral.sh/uv/) installed
- Docker Desktop or another Docker runtime for local Postgres, Qdrant, and MinIO
- A valid `JINA_API_KEY`

## Quick Start

### 1. Clone and enter the project

```bash
cd /Users/jayson/Downloads/Work/groundingDINO_CLIP
```

### 2. Copy the environment file

```bash
cp .env.example .env
```

Then edit `.env` and set at least:

```env
JINA_API_KEY=your_real_jina_api_key
```

The checked-in `.env.example` is aligned with the bundled `docker-compose.yml` defaults:

- Postgres: `postgres/postgres` on `127.0.0.1:5432`
- MinIO: `minioadmin/minioadmin` on `127.0.0.1:9000`
- Qdrant: `127.0.0.1:6333`

### 3. Start local infrastructure

Using the Makefile:

```bash
make infra-up
```

Or directly:

```bash
docker compose up -d
```

This starts:

- Postgres on `127.0.0.1:5432`
- Qdrant on `127.0.0.1:6333`
- MinIO API on `127.0.0.1:9000`
- MinIO console on `127.0.0.1:9001`

### 4. Install dependencies

```bash
uv sync
```

### 5. Run database migrations

Using the Makefile:

```bash
make db-migrate
```

Or directly:

```bash
uv run alembic upgrade head
```

### 6. Start the API server

Using the Makefile:

```bash
make dev
```

Or directly:

```bash
uv run uvicorn app.main:app --reload
```

### 7. Open the API docs

Scalar docs:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

OpenAPI schema:

[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

## First-Run Notes

- The app does not auto-create Postgres tables at startup. Migrations are required.
- The S3 bucket is created automatically on app startup if it does not exist.
- The first detection request may be slow because GroundingDINO weights will be downloaded from Hugging Face.
- Detection runs on CPU only in the current setup, so expect slower inference than GPU.

## Environment Variables

The service reads all config from `.env`.

### Database

- `DATABASE_URL`
  - Recommended format: `postgresql://user:password@host:port/dbname`
  - The app normalizes this to the `psycopg` SQLAlchemy driver internally.
- `DATABASE_MAX_CONNECTIONS`
  - Currently informational only. It is not yet wired into the SQLAlchemy engine config.

### Jina

- `JINA_API_KEY`
- `JINA_EMBEDDING_MODEL`
  - Default: `jina-clip-v2`
- `JINA_EMBEDDINGS_URL`
  - Default: `https://api.jina.ai/v1/embeddings`
- `JINA_BATCH_SIZE`
- `JINA_TIMEOUT_SECONDS`

### Qdrant

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`

### S3 / MinIO

- `S3_ENDPOINT_URL`
  - Local example: `http://127.0.0.1:9000`
- `S3_REGION`
- `S3_BUCKET_NAME`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_USE_SSL`

### Detection

- `GROUNDING_DINO_MODEL_ID`
  - Default: `IDEA-Research/grounding-dino-tiny`
- `GROUNDING_DINO_DEVICE`
  - Default: `cpu`
- `DETECTION_PROMPT`
- `DETECTION_BOX_THRESHOLD`
- `DETECTION_TEXT_THRESHOLD`

### Classification

- `CLASSIFICATION_TOP_K`
- `CLASSIFICATION_MATCH_THRESHOLD`
- `CLASSIFICATION_MARGIN_THRESHOLD`
- `CLASSIFICATION_FALLBACK_TO_FULL_IMAGE`

## API Overview

Base URL:

```text
http://127.0.0.1:8000/api/v1
```

All write APIs currently accept raw JSON, not multipart form data.

Authentication is not implemented yet. Multi-tenant separation is currently done by the `user_id` field in each request.

### `GET /health`

Returns:

```json
{"status":"ok"}
```

### `POST /logos/reference-images`

Enroll one logo class using a list of public image URLs.

Request:

```json
{
  "user_id": "user-1",
  "logo_name": "nike",
  "image_urls": [
    "https://example.com/nike-1.png",
    "https://example.com/nike-2.png",
    "https://example.com/nike-3.png",
    "https://example.com/nike-4.png",
    "https://example.com/nike-5.png"
  ]
}
```

What happens:

- the backend downloads each image URL
- normalizes the image
- uploads it to MinIO/S3
- stores `source_url`, `storage_key`, and `storage_url` in Postgres
- converts the normalized image to a base64 data URL
- calls Jina to get embeddings
- stores the vectors in Qdrant

### `POST /products/detect-logo`

Detect the most likely logo region in a product image.

Request:

```json
{
  "user_id": "user-1",
  "image_url": "https://example.com/product.jpg"
}
```

Response includes:

- `product_id`
- `job_id`
- `found`
- `detection`

### `POST /products/classify-logo`

Detect and classify the logo in a product image.

Request:

```json
{
  "user_id": "user-1",
  "image_url": "https://example.com/product.jpg"
}
```

Response includes:

- detection bounding box
- top predicted logo
- similarity score
- top1-vs-top2 margin
- candidate matches
- `matched=false` if the prediction is below threshold

If the prediction is not confident enough, the API returns:

- `predicted_logo_id: null`
- `predicted_logo_name: null`
- `matched: false`

## Example cURL Commands

Health:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Upload reference images:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/logos/reference-images \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user-1",
    "logo_name": "nike",
    "image_urls": [
      "https://your-public-cdn.com/nike-1.png",
      "https://your-public-cdn.com/nike-2.png",
      "https://your-public-cdn.com/nike-3.png",
      "https://your-public-cdn.com/nike-4.png",
      "https://your-public-cdn.com/nike-5.png"
    ]
  }'
```

Detect logo:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/products/detect-logo \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user-1",
    "image_url": "https://your-public-cdn.com/product.jpg"
  }'
```

Classify logo:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/products/classify-logo \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user-1",
    "image_url": "https://your-public-cdn.com/product.jpg"
  }'
```

## Database Migrations

Create a new migration after changing SQLAlchemy models:

```bash
make db-generate m="add_new_columns"
```

Equivalent direct command:

```bash
uv run alembic revision --autogenerate -m "add_new_columns"
```

Apply migrations:

```bash
make db-migrate
```

Equivalent direct command:

```bash
uv run alembic upgrade head
```

Check current revision:

```bash
make db-current
```

Equivalent direct command:

```bash
uv run alembic current
```

## Local Development Commands

Start infra:

```bash
make infra-up
```

Stop infra:

```bash
make infra-down
```

Run the app:

```bash
make dev
```

Lint:

```bash
uv run ruff check app alembic
```

## How Classification Works

Current classification logic:

1. Detect the most likely logo box with GroundingDINO.
2. Crop the detected logo region.
3. Convert the crop to PNG bytes and embed it with Jina.
4. Query Qdrant for nearest reference vectors filtered by `user_id`.
5. Aggregate nearest hits by `logo_id`.
6. Return the top result only if:
   - score >= `CLASSIFICATION_MATCH_THRESHOLD`
   - top1 - top2 >= `CLASSIFICATION_MARGIN_THRESHOLD`
7. Otherwise return `unknown` behavior with `matched=false`.

If no detection is found and `CLASSIFICATION_FALLBACK_TO_FULL_IMAGE=true`, the classifier falls back to the full image.

## Important Implementation Notes

- Jina image embedding requests are sent as base64 image data URLs.
- The backend stores its own copy of remote images in MinIO/S3, instead of relying on third-party URLs after ingestion.
- The backend stores both the original `source_url` and its own `storage_key` / `storage_url` in Postgres.
- Qdrant collection creation is automatic on the first successful reference upload.
- GroundingDINO is currently configured for CPU only.
- No auth system is implemented yet.

## Troubleshooting

### `ModuleNotFoundError: psycopg2`

Use a normal Postgres URL in `.env` like:

```env
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/logo_service
```

The app converts this internally to the `psycopg` SQLAlchemy driver.

### MinIO startup or bucket errors

Check:

- `S3_ENDPOINT_URL` matches your MinIO endpoint
- `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` are correct
- MinIO is reachable on `127.0.0.1:9000`

For the bundled local stack, the defaults are:

- endpoint: `http://127.0.0.1:9000`
- access key: `minioadmin`
- secret key: `minioadmin`

### Existing database created before Alembic setup

If you created tables before migrations were added, your schema may not match the current models. In local development, the simplest fix is usually to reset the database and run:

```bash
make db-migrate
```

### Slow first detection request

This is expected. The first request downloads the GroundingDINO model.
