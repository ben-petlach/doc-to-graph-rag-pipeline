from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from functools import partial

import anyio
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from neo4j_graphrag.experimental.components.schema import GraphSchema

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
    file_id: str
    filename: str
    status: str


class FileInfo(BaseModel):
    id: str
    name: str
    stage: str
    date_uploaded: str
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

    file_id = registry.set_file_stage(filename=safe_name, stage="uploaded")
    return UploadResponse(file_id=file_id, filename=safe_name, status="uploaded")


@router.get("/documents", tags=["Documents"], response_model=ListFilesResponse)
async def list_documents(request: Request) -> ListFilesResponse:
    settings = _get_settings(request)
    registry = _get_registry(request)

    registry.sync_files_from_disk(data_dir=settings.data_dir)
    files = [
        FileInfo(
            id=r.id,
            name=r.name,
            stage=r.stage,
            date_uploaded=r.date_uploaded,
            error=r.error
        ) 
        for r in registry.list_files() 
        if r.stage != "deleted"
    ]
    files.sort(key=lambda x: x.date_uploaded, reverse=True)
    return ListFilesResponse(files=files)


@router.delete("/documents/{file_id}", tags=["Documents"], response_model=DeleteResponse)
async def delete_document(request: Request, file_id: str) -> DeleteResponse:
    settings = _get_settings(request)
    registry = _get_registry(request)

    # Get file record to find the filename
    file_record = registry.get_file_by_id(file_id=file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_record.stage == "deleted":
        raise HTTPException(status_code=404, detail="File already deleted")

    # Delete the actual files
    raw_path = settings.data_dir / file_record.name
    if raw_path.exists():
        raw_path.unlink()

    out_path = settings.output_dir / f"{Path(file_record.name).stem}.txt"
    if out_path.exists():
        out_path.unlink()

    # Mark as deleted in registry
    registry.delete_file_by_id(file_id=file_id)
    return DeleteResponse(message=f"Deleted {file_record.name}")


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
    
    # Step 1: Extract text from all files
    extracted_texts: list[tuple[str, str]] = []  # [(text, filename), ...]
    
    for fp in files:
        registry.set_file_stage(filename=fp.name, stage="processing")
        try:
            text = await anyio.to_thread.run_sync(
                partial(
                    extract_text_from_file,
                    output_dir=output_dir,
                    write_output=True,
                    force=force_reprocess,
                    ocr_min_confidence=settings.ocr_min_confidence,
                    mistral_api_key=settings.mistral_api_key,
                ),
                fp,
            )
            registry.set_file_stage(filename=fp.name, stage="ocr_complete")
            extracted_texts.append((text, fp.name))
        except Exception as e:
            msg = f"{fp.name}: {e}"
            errors.append(msg)
            registry.set_file_stage(filename=fp.name, stage="failed", error=str(e))

    # Step 2: Generate schema from sample text
    if extracted_texts:
        sample_text = ""
        for text, _ in extracted_texts:
            sample_text += text + "\n\n"
            if len(sample_text) >= settings.kg_schema_sample_size:
                break
        sample_text = sample_text[:settings.kg_schema_sample_size]
        
        try:
            registry.set_file_stage(filename="__schema__", stage="generating_schema")
            schema = await ingestion_service.generate_schema(sample_text=sample_text)
            
            # Save the generated schema
            schema_path = output_dir / "generated_schema.json"
            schema.save(str(schema_path), overwrite=True)
            registry.set_file_stage(filename="__schema__", stage="schema_complete")
        except Exception as e:
            msg = f"Schema generation failed: {e}"
            errors.append(msg)
            registry.set_file_stage(filename="__schema__", stage="failed", error=str(e))
            # Continue without schema (will use LLM auto-extraction)
    
    # Step 3: Ingest all extracted texts with the generated schema
    processed = 0
    for text, filename in extracted_texts:
        registry.set_file_stage(filename=filename, stage="indexing")
        try:
            await ingestion_service.ingest_text(text=text, filename=filename)
            registry.set_file_stage(filename=filename, stage="indexed")
        except Exception as e:
            msg = f"{filename}: {e}"
            errors.append(msg)
            registry.set_file_stage(filename=filename, stage="failed", error=str(e))

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


class SchemaInfo(BaseModel):
    node_types: list[dict[str, Any]]
    relationship_types: list[dict[str, Any]]
    patterns: list[list[str]]


@router.get("/schema", tags=["Schema"], response_model=SchemaInfo)
async def get_schema(request: Request) -> SchemaInfo:
    """Return the LLM-generated graph schema from the last pipeline run."""
    settings = _get_settings(request)
    schema_path = settings.output_dir / "generated_schema.json"
    
    if not schema_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="No schema generated yet. Run the pipeline first."
        )
    
    schema = GraphSchema.from_file(str(schema_path))
    return SchemaInfo(
        node_types=[nt.model_dump() for nt in schema.node_types],
        relationship_types=[rt.model_dump() for rt in schema.relationship_types],
        patterns=list(schema.patterns),
    )


@router.post("/chat/query", tags=["RAG"], response_model=QueryResponse)
async def chat_query(request: Request, payload: QueryRequest) -> QueryResponse:
    graph_qa = _get_graph_qa(request)
    qa = await anyio.to_thread.run_sync(
        partial(graph_qa.ask_question, query=payload.query, top_k=payload.top_k)
    )
    return QueryResponse(answer=qa.answer, sources=qa.sources)

