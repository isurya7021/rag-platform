from uuid import uuid4
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel
)
from app.vector_store.qdrant import QdrantVectorStore
from app.models.chunk import Chunk


embedding_model = SentenceTransformerEmbeddingModel()

vector_store = QdrantVectorStore(
    collection_name="test_documents",
    vector_size=384
)

vector_store.recreate_collection(vector_size=384)


chunks = [
    Chunk(
        id=str(uuid4()),
        content="Redis is an in-memory database used for caching.",
        metadata={
            "document_name": "redis.txt",
            "section": "Introduction"
        }
    ),
    Chunk(
        id=str(uuid4()),
        content="Redis is commonly used as a cache in backend systems.",
        metadata={
            "document_name": "redis.txt",
            "section": "Use Cases"
        }
    ),
    Chunk(
        id=str(uuid4()),
        content="The weather is sunny today.",
        metadata={
            "document_name": "weather.txt",
            "section": "Weather"
        }
    ),
]


# Generate embeddings for all chunks
vectors = embedding_model.embed_batch(
    [chunk.content for chunk in chunks]
)

print("Generated vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))


# Store chunks + vectors in Qdrant
vector_store.upsert(
    chunks=chunks,
    vectors=vectors
)

print("Chunks stored in Qdrant")


# Create embedding for the user's query
query = "What is Redis used for?"

query_vector = embedding_model.embed(query)


# Search Qdrant
results = vector_store.search(
    query_vector=query_vector,
    limit=2
)


print("\nSearch results:")

for result in results:
    print("--------------------")
    print("ID:", result.id)
    print("Content:", result.content)
    print("Score:", result.score)
    print("Metadata:", result.metadata)