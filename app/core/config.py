from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from dotenv import load_dotenv


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: Optional[str]
    openai_api_key: Optional[str]
    mistral_api_key: Optional[str]

    data_dir: Path
    output_dir: Path

    # Retrieval
    neo4j_vector_index: str = "chunkEmbedding"
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4o"
    
    # OCR
    ocr_min_confidence: float = 80.0
    
    # Schema Generation
    kg_max_node_types: int = 10
    kg_max_relationship_types: int = 20
    kg_schema_sample_size: int = 8000


def load_settings() -> Settings:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE") or None
    openai_api_key = os.getenv("OPENAI_API_KEY") or None
    mistral_api_key = os.getenv("MISTRAL_API_KEY") or None

    if not neo4j_uri:
        raise RuntimeError("Missing required env var: NEO4J_URI")
    if not neo4j_username:
        raise RuntimeError("Missing required env var: NEO4J_USERNAME")
    if not neo4j_password:
        raise RuntimeError("Missing required env var: NEO4J_PASSWORD")

    data_dir = PROJECT_ROOT / "pipeline" / "data"
    output_dir = PROJECT_ROOT / "pipeline" / "output"
    
    # OCR settings
    ocr_min_confidence = float(os.getenv("OCR_MIN_CONFIDENCE", "80.0"))
    
    # Schema generation settings
    kg_max_node_types = int(os.getenv("KG_MAX_NODE_TYPES", "10"))
    kg_max_relationship_types = int(os.getenv("KG_MAX_RELATIONSHIP_TYPES", "20"))
    kg_schema_sample_size = int(os.getenv("KG_SCHEMA_SAMPLE_SIZE", "8000"))

    return Settings(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        openai_api_key=openai_api_key,
        mistral_api_key=mistral_api_key,
        data_dir=data_dir,
        output_dir=output_dir,
        ocr_min_confidence=ocr_min_confidence,
        kg_max_node_types=kg_max_node_types,
        kg_max_relationship_types=kg_max_relationship_types,
        kg_schema_sample_size=kg_schema_sample_size,
    )

