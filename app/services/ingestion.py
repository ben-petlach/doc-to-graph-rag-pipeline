from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from neo4j import Driver
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class IngestionResult:
    filename: str
    result: Any


class IngestionService:
    def __init__(self, *, driver: Driver, settings: Settings):
        self._driver = driver
        self._settings = settings

        llm = OpenAILLM(
            model_name=settings.llm_model,
            model_params={
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )

        embedder = OpenAIEmbeddings(model=settings.embedding_model)
        text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

        self._kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            neo4j_database=settings.neo4j_database,
            embedder=embedder,
            from_pdf=False,
            text_splitter=text_splitter,
        )

    async def ingest_text(self, *, text: str, filename: str) -> IngestionResult:
        """
        Ingest text into Neo4j via SimpleKGPipeline.

        `filename` is carried through for API/task bookkeeping.
        """
        pipeline_result = await self._kg_builder.run_async(text=text)
        return IngestionResult(filename=filename, result=pipeline_result.result)

