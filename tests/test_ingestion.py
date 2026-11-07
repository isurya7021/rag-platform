from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from app.ingestion.service import IngestionService
from app.vector_store.qdrant import QdrantVectorStore
from app.document_store.postgres import PostgresDocumentStore


embedding_model = SentenceTransformerEmbeddingModel()

vector_store = QdrantVectorStore(
    collection_name="rag_documents",
    vector_size=384,
)

document_store = PostgresDocumentStore()

ingestion_service = IngestionService(
    embedding_model=embedding_model,
    vector_store=vector_store,
    document_store=document_store,
)

file_path = "documents/sample.txt"

chunks_created = ingestion_service.ingest(file_path)

print(f"Successfully ingested {chunks_created} chunks.")