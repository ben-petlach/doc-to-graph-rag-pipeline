#NOTE: the following code was added to rate_limit.py in the neo4j-graphrag.llm package to help with rate limit issue:
"""
# Default rate limit handler instance
DEFAULT_RATE_LIMIT_HANDLER = RetryRateLimitHandler(
    max_attempts=6,
    min_wait=2.0,
    max_wait=120.0,
    multiplier=2.0,
    jitter=True,
)
"""
# hopefully in production we'll have better rate limits.

# another potential solution is to use a "cheaper" embedding model

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.schema import GraphSchema, SchemaFromTextExtractor
from neo4j_graphrag.generation.prompts import PromptTemplate

from speed_analyzer import tracker

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("./pipeline/output")
MAX_NODE_TYPES = int(os.getenv("KG_MAX_NODE_TYPES", "10"))
MAX_RELATIONSHIP_TYPES = int(os.getenv("KG_MAX_RELATIONSHIP_TYPES", "20"))
# Max chars to feed the schema extractor (a representative sample of the corpus)
SCHEMA_SAMPLE_SIZE = int(os.getenv("KG_SCHEMA_SAMPLE_SIZE", "8000"))

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100) #chunk overlap helps to maintain context across chunks, which can improve the quality of the extracted entities and relationships, especially when they span across chunk boundaries.


# ---------------------------------------------------------------------------
# Constrained schema extraction via LLM
# ---------------------------------------------------------------------------

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


async def generate_schema_from_text(sample_text: str) -> GraphSchema:
    """
    Use the LLM to generate a constrained GraphSchema from a representative
    sample of the corpus. The schema is anchored to MAX_NODE_TYPES node types
    and MAX_RELATIONSHIP_TYPES relationship types.
    """
    prompt_template = ConstrainedSchemaExtractionTemplate(
        max_node_types=MAX_NODE_TYPES,
        max_relationship_types=MAX_RELATIONSHIP_TYPES,
    )

    # Use a separate LLM instance without forced json_object response_format
    # since SchemaFromTextExtractor handles JSON parsing itself
    schema_llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    )

    extractor = SchemaFromTextExtractor(
        llm=schema_llm,
        prompt_template=prompt_template,
    )

    schema = await extractor.run(text=sample_text)

    # Enforce strict adherence — no additional types beyond what was generated
    schema_dict = schema.model_dump(mode="json")
    schema_dict["additional_node_types"] = False
    schema_dict["additional_relationship_types"] = False
    schema_dict["additional_patterns"] = False
    schema = GraphSchema.model_validate(schema_dict)

    logging.info(
        f"LLM-generated schema: "
        f"{len(schema.node_types)} node types, "
        f"{len(schema.relationship_types)} relationship types, "
        f"{len(schema.patterns)} patterns"
    )
    return schema


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------

# Collect all text files and build a representative sample for schema extraction
text_files = sorted([
    p for p in DATA_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".txt"
])

sample_text = ""
for text_file in text_files:
    with open(text_file, "r", encoding="utf-8") as f:
        sample_text += f.read() + "\n\n"
    if len(sample_text) >= SCHEMA_SAMPLE_SIZE:
        break
sample_text = sample_text[:SCHEMA_SAMPLE_SIZE]

# Generate schema from the sample
logging.info(f"Generating schema from {len(sample_text)} chars of sample text "
             f"(target: {MAX_NODE_TYPES} node types, {MAX_RELATIONSHIP_TYPES} relationship types)...")

tracker.start("schema_generation", 0, "schema_extraction")
schema = asyncio.run(generate_schema_from_text(sample_text))
elapsed = tracker.stop()
logging.info(f"Schema generation took {elapsed:.2f}s")

# Save the generated schema for inspection/reuse
GENERATED_SCHEMA_PATH = DATA_DIR / "generated_schema.json"
schema.save(str(GENERATED_SCHEMA_PATH), overwrite=True)
logging.info(f"Generated schema saved to {GENERATED_SCHEMA_PATH}")

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    embedder=embedder,
    from_pdf=False,
    text_splitter=text_splitter,
    schema=schema,
)


# ---------------------------------------------------------------------------
# Run KG extraction with the generated schema
# ---------------------------------------------------------------------------

for text_file in text_files:
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Load page count from companion metadata (written by ocr_preprocessor)
    meta_path = text_file.with_suffix(".meta.json")
    page_count = 1
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
                page_count = meta.get("page_count", 1)
        except Exception:
            pass

    tracker.start(text_file.name, 0, "kg_extraction")
    result = asyncio.run(kg_builder.run_async(text=text))
    elapsed = tracker.stop()

    # Record approximate per-page KG time
    if page_count > 1:
        per_page = elapsed / page_count
        for pg in range(1, page_count + 1):
            tracker.record(text_file.name, pg, "kg_extraction_per_page", per_page)

    print(text_file.name, result.result)


# Print and save speed report
print("\n" + tracker.summary())
tracker.save(DATA_DIR / "speed_report_kg.csv")

neo4j_driver.close()