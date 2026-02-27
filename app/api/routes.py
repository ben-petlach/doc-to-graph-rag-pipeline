from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from functools import partial

import anyio
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.file_utils import sanitize_filename
from app.core.task_registry import InMemoryRegistry
from app.services.ingestion import IngestionService
from app.services.ocr import SUPPORTED_EXTENSIONS, extract_text_from_file
from app.services.retrieval import GraphQA


router = APIRouter(prefix="/api/v1")

def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _get_registry(request: Request) -> InMemoryRegistry:
    return request.app.state.registry


def _get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def _get_graph_qa(request: Request) -> GraphQA:
    return request.app.state.graph_qa


class UploadResponse(BaseModel):
    filename: str
    status: str


class FileInfo(BaseModel):
    name: str
    stage: str
    error: Optional[str] = None


class ListFilesResponse(BaseModel):
    files: list[FileInfo]


class DeleteResponse(BaseModel):
    message: str


class ProcessRequest(BaseModel):
    force_reprocess: bool = False


class ProcessResponse(BaseModel):
    task_id: str
    message: str


class StatusResponse(BaseModel):
    status: str
    progress: str
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


def _validate_path_param_filename(filename: str) -> str:
    # Reject attempts to pass paths; allow plain filenames only.
    base = Path(filename).name
    if base != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return base


@router.post("/documents/upload", tags=["Documents"], response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    settings = _get_settings(request)
    registry = _get_registry(request)

    safe_name = sanitize_filename(file.filename or "upload")
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dest_path = settings.data_dir / safe_name
    contents = await file.read()
    dest_path.write_bytes(contents)

    registry.set_file_stage(filename=safe_name, stage="uploaded")
    return UploadResponse(filename=safe_name, status="uploaded")


@router.get("/documents", tags=["Documents"], response_model=ListFilesResponse)
async def list_documents(request: Request) -> ListFilesResponse:
    settings = _get_settings(request)
    registry = _get_registry(request)

    registry.sync_files_from_disk(data_dir=settings.data_dir)
    files = [FileInfo(name=r.name, stage=r.stage, error=r.error) for r in registry.list_files()]
    files.sort(key=lambda x: x.name.lower())
    return ListFilesResponse(files=files)


@router.delete("/documents/{filename}", tags=["Documents"], response_model=DeleteResponse)
async def delete_document(request: Request, filename: str) -> DeleteResponse:
    settings = _get_settings(request)
    registry = _get_registry(request)

    name = _validate_path_param_filename(filename)
    raw_path = settings.data_dir / name
    if raw_path.exists():
        raw_path.unlink()

    out_path = settings.output_dir / f"{Path(name).stem}.txt"
    if out_path.exists():
        out_path.unlink()

    registry.delete_file_record(filename=name)
    return DeleteResponse(message="Deleted")


async def _run_pipeline_job(
    *,
    task_id: str,
    settings: Settings,
    registry: InMemoryRegistry,
    ingestion_service: IngestionService,
    force_reprocess: bool,
) -> None:
    data_dir = settings.data_dir
    output_dir = settings.output_dir

    files = [
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda p: p.name.lower())

    registry.update_task(task_id=task_id, status="processing", processed_files=0, total_files=len(files))

    errors: list[str] = []
    processed = 0

    for fp in files:
        registry.set_file_stage(filename=fp.name, stage="processing")
        try:
            text = await anyio.to_thread.run_sync(
                partial(
                    extract_text_from_file,
                    output_dir=output_dir,
                    write_output=True,
                    force=force_reprocess,
                ),
                fp,
            )
            registry.set_file_stage(filename=fp.name, stage="ocr_complete")

            await ingestion_service.ingest_text(text=text, filename=fp.name)
            registry.set_file_stage(filename=fp.name, stage="indexed")
        except Exception as e:
            msg = f"{fp.name}: {e}"
            errors.append(msg)
            registry.set_file_stage(filename=fp.name, stage="failed", error=str(e))

        processed += 1
        registry.update_task(task_id=task_id, processed_files=processed)

    if errors:
        registry.update_task(task_id=task_id, status="failed", error="; ".join(errors[:5]))
    else:
        registry.update_task(task_id=task_id, status="completed", error=None)


@router.post("/pipeline/process", tags=["Pipeline"], response_model=ProcessResponse)
async def process_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: ProcessRequest,
) -> ProcessResponse:
    settings = _get_settings(request)
    registry = _get_registry(request)
    ingestion_service = _get_ingestion_service(request)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        p
        for p in settings.data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    task_id = uuid4().hex
    registry.create_task(task_id=task_id, total_files=len(files))

    background_tasks.add_task(
        _run_pipeline_job,
        task_id=task_id,
        settings=settings,
        registry=registry,
        ingestion_service=ingestion_service,
        force_reprocess=payload.force_reprocess,
    )

    return ProcessResponse(task_id=task_id, message="Processing started")


@router.get("/pipeline/status/{task_id}", tags=["Pipeline"], response_model=StatusResponse)
async def pipeline_status(request: Request, task_id: str) -> StatusResponse:
    registry = _get_registry(request)
    rec = registry.get_task(task_id=task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id (server restart clears in-memory status)")
    return StatusResponse(status=rec.status, progress=rec.progress, error=rec.error)


@router.post("/chat/query", tags=["RAG"], response_model=QueryResponse)
async def chat_query(request: Request, payload: QueryRequest) -> QueryResponse:
    graph_qa = _get_graph_qa(request)
    qa = await anyio.to_thread.run_sync(
        partial(graph_qa.ask_question, query=payload.query, top_k=payload.top_k)
    )
    return QueryResponse(answer=qa.answer, sources=qa.sources)

