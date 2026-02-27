# Document-to-Graph Pipeline

This project builds a knowledge graph from documents and exposes a RAG-style query interface on top of Neo4j.

## FastAPI service

Run the API (from the project root) after installing dependencies and setting up `.env`:

```bash
uvicorn main:app --reload
```

Interactive API docs are available at:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Key endpoints

- **Upload document**
  - **POST** `/api/v1/documents/upload`
  - Body: `multipart/form-data` with `file`
  - Saves the file to `pipeline/data/` and marks it as `uploaded`.

- **List documents**
  - **GET** `/api/v1/documents`
  - Returns all known files and their current stage (e.g. `uploaded`, `ocr_complete`, `indexed`, `failed`).

- **Delete document**
  - **DELETE** `/api/v1/documents/{filename}`
  - Removes the file from `pipeline/data/` and any derived OCR output in `pipeline/output/`.

- **Trigger pipeline (OCR → KG ingestion)**
  - **POST** `/api/v1/pipeline/process`
  - JSON body: `{"force_reprocess": false}`
  - Starts a background job that OCRs and ingests all supported files in `pipeline/data/`, and returns a `task_id` immediately.

- **Check pipeline status**
  - **GET** `/api/v1/pipeline/status/{task_id}`
  - Returns overall status and progress for the background job.

- **RAG query**
  - **POST** `/api/v1/chat/query`
  - JSON body: `{"query": "...", "top_k": 5}`
  - Returns an answer plus the underlying sources from the Neo4j graph.

## Legacy scripts

- `pipeline/ocr_preprocessor.py` – OCR PDFs/images in `pipeline/data/` into `.txt` files.
- `pipeline/kg_builder.py` – Ingest `.txt` files, build embeddings, and write the graph to Neo4j.
- `pipeline/vector_cypher_rag.py` – Run a RAG query over the graph and print the answer plus context.

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

