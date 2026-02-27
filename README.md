# Document-to-Graph Pipeline

This project builds a knowledge graph from documents and exposes a simple RAG-style query interface on top of Neo4j.

## Scripts

- `pipeline/ocr_preprocessor.py` – OCR PDFs/images in `pipeline/data/` into `.txt` files.
- `pipeline/kg_builder.py` – Ingest `.txt` files, build embeddings, and write the graph to Neo4j.
- `pipeline/vector_cypher_rag.py` – Run a RAG query over the graph and print the answer plus context.

Each script is meant to be run directly (for example, `python pipeline/kg_builder.py`) and stays small and self-contained.

## Environment

Set the following in your `.env`:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `OPENAI_API_KEY`

## One-Time Neo4j Index

Before running the pipeline, create a vector index in Neo4j (run once per database), for example in Neo4j Browser or `cypher-shell`:

```cypher
CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
FOR (n:Chunk) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

## Run Order

1. Drop PDFs/images into `pipeline/data/`.
2. Run `python pipeline/ocr_preprocessor.py` to generate `.txt` files.
3. Run `python pipeline/kg_builder.py` to build the graph.
4. Run `python pipeline/vector_cypher_rag.py` to query the graph.

