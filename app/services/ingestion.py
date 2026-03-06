from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from neo4j import Driver
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.schema import GraphSchema, SchemaFromTextExtractor
from neo4j_graphrag.generation.prompts import PromptTemplate
from neo4j_graphrag.llm import OpenAILLM

from app.core.config import Settings


logger = logging.getLogger(__name__)


CONSTRAINED_SCHEMA_PROMPT = """
You are a top-tier algorithm designed for extracting a labeled property graph schema in
structured formats.

Generate a generalized graph schema based on the input text. Identify key node types,
their relationship types, and property types.

HARD CONSTRAINTS:
- Return EXACTLY {max_node_types} node types (no more, no fewer). Pick the {max_node_types} most
  important and distinct entity categories present in the text.
- Return EXACTLY {max_relationship_types} relationship types (no more, no fewer). Pick the
  {max_relationship_types} most important and distinct relationships between those entities.
- These limits help "anchor" the schema and keep the knowledge graph manageable.

IMPORTANT RULES:
1. Return only abstract schema information, not concrete instances.
2. Use singular PascalCase labels for node types (e.g., Person, Company, Product).
3. Use UPPER_SNAKE_CASE labels for relationship types (e.g., WORKS_FOR, MANAGES).
4. Include property definitions only when the type can be confidently inferred, otherwise omit them.
5. When defining patterns, ensure every node label and relationship label mentioned exists
   in your lists of node types and relationship types.
6. Do not create node types that aren't clearly mentioned in the text.
7. Every relationship type MUST appear in at least one pattern.

Accepted property types are: BOOLEAN, DATE, DURATION, FLOAT, INTEGER, LIST,
LOCAL_DATETIME, LOCAL_TIME, POINT, STRING, ZONED_DATETIME, ZONED_TIME.

Return a valid JSON object that follows this precise structure:
{{{{
  "node_types": [
    {{{{
      "label": "Person",
      "properties": [
        {{{{
          "name": "name",
          "type": "STRING"
        }}}}
      ]
    }}}},
    ...
  ],
  "relationship_types": [
    {{{{
      "label": "WORKS_FOR"
    }}}},
    ...
  ],
  "patterns": [
    ["Person", "WORKS_FOR", "Company"],
    ...
  ]
}}}}

Examples:
{{examples}}

Input text:
{{text}}
"""


class ConstrainedSchemaExtractionTemplate(PromptTemplate):
    """Prompt template that instructs the LLM to produce exactly N node types
    and M relationship types."""

    DEFAULT_TEMPLATE = CONSTRAINED_SCHEMA_PROMPT
    EXPECTED_INPUTS = ["text"]

    def __init__(self, max_node_types: int = 10, max_relationship_types: int = 20):
        # Pre-fill the constraint placeholders so they're baked into the template
        template = CONSTRAINED_SCHEMA_PROMPT.format(
            max_node_types=max_node_types,
            max_relationship_types=max_relationship_types,
        )
        super().__init__(template=template, expected_inputs=["text"])

    def format(self, text: str = "", examples: str = "", **kwargs) -> str:
        return super().format(text=text, examples=examples)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    filename: str
    result: Any


class IngestionService:
    def __init__(self, *, driver: Driver, settings: Settings):
        self._driver = driver
        self._settings = settings
        self._llm = OpenAILLM(
            model_name=settings.llm_model,
            model_params={
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        self._embedder = OpenAIEmbeddings(model=settings.embedding_model)
        self._text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)
        
        # KG builder will be created with or without schema as needed
        self._kg_builder: Optional[SimpleKGPipeline] = None
        self._current_schema: Optional[GraphSchema] = None

    async def generate_schema(
        self, 
        sample_text: str,
    ) -> GraphSchema:
        """
        Generate a constrained GraphSchema from a sample of text.
        Uses the LLM to identify the most important node and relationship types,
        constrained to the limits defined in settings.
        """
        prompt_template = ConstrainedSchemaExtractionTemplate(
            max_node_types=self._settings.kg_max_node_types,
            max_relationship_types=self._settings.kg_max_relationship_types,
        )

        extractor = SchemaFromTextExtractor(
            llm=self._llm,
            prompt_template=prompt_template,
        )

        schema = await extractor.run(text=sample_text)

        # Enforce strict adherence — no additional types beyond what was generated
        schema_dict = schema.model_dump(mode="json")
        schema_dict["additional_node_types"] = False
        schema_dict["additional_relationship_types"] = False
        schema_dict["additional_patterns"] = False
        schema = GraphSchema.model_validate(schema_dict)

        logger.info(
            f"Generated schema: "
            f"{len(schema.node_types)} node types, "
            f"{len(schema.relationship_types)} relationship types, "
            f"{len(schema.patterns)} patterns"
        )
        
        self._current_schema = schema
        return schema

    def set_schema(self, schema: GraphSchema) -> None:
        """Set the schema to use for subsequent ingestion operations."""
        self._current_schema = schema
        # Reset kg_builder so it will be recreated with the new schema
        self._kg_builder = None

    def _ensure_kg_builder(self) -> SimpleKGPipeline:
        """Lazily create or return the KG builder with current schema."""
        if self._kg_builder is None:
            kwargs = dict(
                llm=self._llm,
                driver=self._driver,
                neo4j_database=self._settings.neo4j_database,
                embedder=self._embedder,
                from_pdf=False,
                text_splitter=self._text_splitter,
            )
            
            if self._current_schema is not None:
                kwargs["schema"] = self._current_schema
                logger.info("Creating KG builder with constrained schema")
            else:
                logger.info("Creating KG builder with LLM auto-extraction (no schema)")
            
            self._kg_builder = SimpleKGPipeline(**kwargs)
        
        return self._kg_builder

    async def ingest_text(self, *, text: str, filename: str) -> IngestionResult:
        """
        Ingest text into Neo4j via SimpleKGPipeline.

        Uses the current schema if one has been set via generate_schema() or set_schema().
        Otherwise falls back to LLM auto-extraction.

        `filename` is carried through for API/task bookkeeping.
        """
        kg_builder = self._ensure_kg_builder()
        pipeline_result = await kg_builder.run_async(text=text)
        return IngestionResult(filename=filename, result=pipeline_result.result)

