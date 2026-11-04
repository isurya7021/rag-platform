from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from app.ingestion.service import IngestionService
from app.vector_store.qdrant import QdrantVectorStore


embedding_model = SentenceTransformerEmbeddingModel()

vector_store = QdrantVectorStore(
    collection_name="rag_documents",
    vector_size=384,
)

ingestion_service = IngestionService(
    embedding_model=embedding_model,
    vector_store=vector_store,
)

file_path = "documents/sample.txt"

chunks_created = ingestion_service.ingest(file_path)

print(f"Successfully ingested {chunks_created} chunks.")