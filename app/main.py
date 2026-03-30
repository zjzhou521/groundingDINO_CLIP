import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from scalar_fastapi import AgentScalarConfig, get_scalar_api_reference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from app.api.routes import router
from app.core.config import settings
from app.dependencies import (
    get_embedding_service,
    get_network_service,
    get_object_storage_service,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_object_storage_service().ensure_bucket()
    yield
    get_embedding_service().close()
    get_network_service().close()


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.include_router(router, prefix=settings.API_V1_PREFIX)


@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=f"{settings.APP_NAME} API Reference",
        dark_mode=False,
        hide_dark_mode_toggle=False,
        with_default_fonts=True,
        agent=AgentScalarConfig(disabled=True),
    )
