from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import Driver
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever

from app.core.config import Settings


RETRIEVAL_QUERY = """
MATCH (node)-[:FROM_DOCUMENT]->(d)
RETURN
    node.text as text, score,
    d.path as source_path,
    collect {
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            [l IN labels(entity) WHERE l <> 'Chunk' AND l <> '__Entity__' | l][0],
            entity.name,
            entity.type,
            entity.description,
            type(r),
            [l IN labels(other) WHERE l <> 'Chunk' AND l <> '__Entity__' | l][0],
            other.name,
            other.type,
            other.description
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""


@dataclass(frozen=True, slots=True)
class QAResponse:
    answer: str
    sources: list[dict[str, Any]]


class GraphQA:
    def __init__(self, *, driver: Driver, settings: Settings):
        embedder = OpenAIEmbeddings(model=settings.embedding_model)

        retriever = VectorCypherRetriever(
            driver,
            neo4j_database=settings.neo4j_database,
            index_name=settings.neo4j_vector_index,
            embedder=embedder,
            retrieval_query=RETRIEVAL_QUERY,
        )

        llm = OpenAILLM(model_name=settings.llm_model)
        self._rag = GraphRAG(retriever=retriever, llm=llm)

    def ask_question(self, *, query: str, top_k: int) -> QAResponse:
        response = self._rag.search(
            query_text=query,
            retriever_config={"top_k": top_k},
            return_context=True,
        )

        sources: list[dict[str, Any]] = []
        items = getattr(getattr(response, "retriever_result", None), "items", None)
        if isinstance(items, list):
            for item in items:
                if hasattr(item, "model_dump"):
                    sources.append(item.model_dump())
                elif hasattr(item, "dict"):
                    sources.append(item.dict())  # type: ignore[call-arg]
                elif isinstance(item, dict):
                    sources.append(item)
                else:
                    sources.append({"item": repr(item)})

        return QAResponse(answer=getattr(response, "answer", ""), sources=sources)

