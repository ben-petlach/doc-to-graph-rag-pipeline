from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import Settings, load_settings
from app.core.database import close_neo4j_driver, create_neo4j_driver
from app.core.task_registry import InMemoryRegistry
from app.services.ingestion import IngestionService
from app.services.retrieval import GraphQA


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    driver = create_neo4j_driver(settings)

    registry = InMemoryRegistry()
    graph_qa = GraphQA(driver=driver, settings=settings)
    ingestion_service = IngestionService(driver=driver, settings=settings)

    app.state.settings = settings
    app.state.neo4j_driver = driver
    app.state.registry = registry
    app.state.graph_qa = graph_qa
    app.state.ingestion_service = ingestion_service

    try:
        yield
    finally:
        close_neo4j_driver(app.state.neo4j_driver)


app = FastAPI(title="Document-to-Graph RAG Pipeline", version="0.1.0", lifespan=lifespan)
app.include_router(api_router)

