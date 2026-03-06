[x] be able to read handwritten text. Proposal: if below certain confidence score, send to LLM to extract
  - Implemented: per-word Tesseract confidence via `image_to_data`. If avg confidence < `OCR_MIN_CONFIDENCE` (80), falls back to Mistral OCR 3.0 (`mistral-ocr-latest`).
  - See `pipeline/ocr_preprocessor.py` — `ocr_image_with_confidence()`, `mistral_ocr_image()`, `mistral_ocr_pdf()`
[x] define schema beforehand. Compare that vs. default extractor vs. providing schema from JSON file
 <!-- - to optimize, you should be able to define them yourself and see how the LLM compares to it. -->
 <!-- - that includes node features and edge features -->
 - Regarding limitations: can tell it to have 10 node types, and 20 edge types to "anchor" it. But it shouldn't be too much more bloated than that due to LLM restrictions
  - Implemented: LLM reads a sample of the corpus and generates a `GraphSchema` constrained to 10 node types and 20 relationship types (configurable via `KG_MAX_NODE_TYPES` / `KG_MAX_RELATIONSHIP_TYPES` env vars). Schema is enforced strictly (`additional_node_types=False`). Generated schema saved to `pipeline/output/generated_schema.json` for inspection. See `pipeline/kg_builder.py`.

[x] speed analysis / page
  - Implemented: `pipeline/speed_analyzer.py` — `SpeedTracker` class tracks per-page, per-stage timing. OCR writes `speed_report_ocr.csv`, KG builder writes `speed_report_kg.csv` to `pipeline/output/`.
<!-- [ ] test on receipts, invoices, parts of novels, scientific papers, etc. to see how it performs on different types of documents -->