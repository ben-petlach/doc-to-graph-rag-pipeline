import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

#TODO: update the non-determenistic attributes coming from the LLM:
# Define retrieval query
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d)
RETURN
    node.text as text, score,
    d.path as source_path,
    collect { 
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            labels(entity)[2], 
            entity.name, 
            entity.type, 
            entity.description, 
            type(r), 
            labels(other)[2], 
            other.name, 
            other.type, 
            other.description
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""

# Create retriever
retriever = VectorCypherRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="chunkEmbedding", #Configure based on the name of the index you created in Neo4j
    embedder=embedder,
    retrieval_query=retrieval_query,
)

#  Create the LLM
llm = OpenAILLM(model_name="gpt-4o")

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
query_text = "What are proposed solutions for this challenge?"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)

# Close the database connection
driver.close()
