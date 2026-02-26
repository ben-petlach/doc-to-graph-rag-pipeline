# Project Plan: Document-to-Graph Knowledge Extraction Pipeline

## Folder Structure

```
root/
├── .env
├── pipeline/
│   ├── data/               # input documents (PDFs, text files)
│   ├── kg_builder.py       # builds the knowledge graph (existing)
│   ├── vector_cypher_rag.py # RAG pipeline (existing)
│   ├── ocr_preprocessor.py # NEW: converts scanned docs to text/PDF
│   └── setup_indexes.py    # NEW: creates Neo4j vector index
```

---

## The Three Stages

### Stage 1 — Document Ingestion & Preprocessing

Your existing `kg_builder.py` already handles standard PDFs via `from_pdf=True`. The only gap is **scanned documents** that need OCR. Add a lightweight preprocessing step before the pipeline runs.

**New file: `ocr_preprocessor.py`**

Use `pytesseract` + `pdf2image` to convert scanned PDFs or images into text-searchable PDFs (or plain `.txt` files). The output drops into `pipeline/data/` just like any other document, so the rest of the pipeline is untouched. The simple decision tree is: if the document is already a native PDF, pass it straight through. If it's a scanned PDF or image, run OCR and save a cleaned version.

**Dependencies to add:** `pytesseract`, `pdf2image`, `Pillow`

---

### Stage 2 — Knowledge Graph Construction

This is almost entirely covered by your existing `kg_builder.py`. A few intentional design decisions worth locking in:

**Keep the schema general-purpose.** Your current `SimpleKGPipeline` has no schema passed in, which means the LLM freely identifies entity types. This is the right call for a general document intelligence system — adding a fixed schema would limit it to specific domains.

**One index, set it up once.** Create `setup_indexes.py` that runs the following Cypher once against your database:

```cypher
CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
FOR (n:Chunk) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

This file only needs to be run one time on a fresh database. It doesn't need to be part of the ingestion loop.

**Processing multiple documents.** Wrap the `kg_builder.run_async()` call in a simple loop over `pipeline/data/` so you can drop in multiple files and process them all in one run. The `SimpleKGPipeline` already tags chunks with their source document, so provenance is preserved automatically.

---

### Stage 3 — RAG Query Interface

Your `vector_cypher_rag.py` is nearly complete. The main thing flagged with `#TODO` in your existing code is the non-deterministic label index (`labels(entity)[2]`). This is fragile — it assumes a specific label ordering that the LLM may not always produce consistently.

**The fix:** Replace `labels(entity)[2]` with a more defensive pattern. Use `[l IN labels(entity) WHERE l <> 'Chunk' AND l <> '__Entity__' | l][0]` to dynamically grab the meaningful label regardless of ordering. Apply this in both entity and `other` positions in the retrieval query.

Beyond that fix, `vector_cypher_rag.py` already covers:
- Vector similarity search over chunks
- Cypher traversal to pull associated entities and relationships
- LLM response generation via `GraphRAG`
- Source provenance via `d.path`

The `query_text` is the only thing that changes per user query, so you can expose this as a simple function or CLI argument.

---

## What Each File Is Responsible For

| File | Responsibility |
|---|---|
| `ocr_preprocessor.py` | Convert scanned/image docs to processable format |
| `setup_indexes.py` | One-time Neo4j vector index creation |
| `kg_builder.py` | Ingest docs → chunk → embed → extract entities → write graph |
| `vector_cypher_rag.py` | Query graph → retrieve context → generate answer |

---

## Key Design Decisions (Justified)

**No custom schema** — keeps the pipeline document-agnostic. Any domain works without reconfiguration.

**`text-embedding-ada-002` for all embeddings** — your instruction doc explicitly notes the same model must be used for both indexing and querying. This is already consistent across both files, just don't change it in one place without the other.

**`chunk_size=500, overlap=100`** — already in your builder. This is a reasonable default for entity extraction quality vs. granularity. No need to change it unless extraction quality is poor on your specific documents.

**`gpt-4o` for extraction and generation** — already set. This matters most for extraction quality; using a weaker model there degrades the whole graph.

**Rate limit handler** — the note in your `kg_builder.py` comment about the custom `RetryRateLimitHandler` in `rate_limit.py` is important. Make sure that patch stays in place for any production or bulk ingestion run.

---

## Implementation Order

1. Run `setup_indexes.py` once on your Neo4j instance
2. Drop documents into `pipeline/data/`
3. Run `ocr_preprocessor.py` on any scanned files (outputs cleaned versions back to `data/`)
4. Run `kg_builder.py` to populate the graph
5. Run `vector_cypher_rag.py` with your query to get a grounded answer

That's the full pipeline end-to-end. No orchestration framework needed — the linear run order is simple enough to execute manually or wrap in a single `main.py` entry point if desired.