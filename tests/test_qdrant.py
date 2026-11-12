
from uuid import uuid4

from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from app.vector_store.qdrant import QdrantVectorStore
from app.models.chunk import Chunk


def main():
    embedding_model = SentenceTransformerEmbeddingModel()

    vector_store = QdrantVectorStore(
        collection_name="test_documents",
        vector_size=384,
    )

    vector_store.recreate_collection(vector_size=384)

    # Separate documents.
    redis_document_id = str(uuid4())
    weather_document_id = str(uuid4())

    # Separate ingestion generations.
    redis_generation_id = str(uuid4())
    weather_generation_id = str(uuid4())

    chunks = [
        Chunk(
            id=str(uuid4()),
            content="Redis is an in-memory database used for caching.",
            metadata={
                "document_id": redis_document_id,
                "document_name": "redis.txt",
                "document_hash": "redis-test-hash",
                "chunk_index": 0,
                "section": "Introduction",
            },
        ),
        Chunk(
            id=str(uuid4()),
            content="Redis is commonly used as a cache in backend systems.",
            metadata={
                "document_id": redis_document_id,
                "document_name": "redis.txt",
                "document_hash": "redis-test-hash",
                "chunk_index": 1,
                "section": "Use Cases",
            },
        ),
        Chunk(
            id=str(uuid4()),
            content="The weather is sunny today.",
            metadata={
                "document_id": weather_document_id,
                "document_name": "weather.txt",
                "document_hash": "weather-test-hash",
                "chunk_index": 0,
                "section": "Weather",
            },
        ),
    ]

    # Generate embeddings for all chunks.
    vectors = embedding_model.embed_batch(
        [chunk.content for chunk in chunks]
    )

    print("Generated vectors:", len(vectors))
    print("Vector dimension:", len(vectors[0]))

    # Store Redis document chunks under its generation.
    redis_chunks = chunks[:2]
    redis_vectors = vectors[:2]

    vector_store.upsert(
        chunks=redis_chunks,
        vectors=redis_vectors,
        generation_id=redis_generation_id,
    )

    # Store weather document chunks under its generation.
    weather_chunks = chunks[2:]
    weather_vectors = vectors[2:]

    vector_store.upsert(
        chunks=weather_chunks,
        vectors=weather_vectors,
        generation_id=weather_generation_id,
    )

    print("Chunks stored in Qdrant")

    # Create embedding for the user's query.
    query = "What is Redis used for?"
    query_vector = embedding_model.embed(query)

    # Search Qdrant.
    results = vector_store.search(
        query_vector=query_vector,
        limit=2,
    )

    print("\nSearch results:")

    for result in results:
        print("--------------------")
        print("ID:", result.id)
        print("Content:", result.content)
        print("Score:", result.score)
        print("Metadata:", result.metadata)


if __name__ == "__main__":
    main()
