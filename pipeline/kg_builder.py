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

import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter;

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

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_pdf=True,
    text_splitter=text_splitter
)

pdf_file = "./pipeline/data/report.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
