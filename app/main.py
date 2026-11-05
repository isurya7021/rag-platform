from fastapi import FastAPI

from app.api.routes import router
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from app.generation.gemini_generator import GeminiGenerator
from app.ingestion.service import IngestionService
from app.rag.service import RAGService
from app.vector_store.qdrant import QdrantVectorStore
from app.document_store.qdrant import QdrantDocumentStore


def create_app() -> FastAPI:

    app = FastAPI(
        title="RAG Platform",
        description="Document ingestion and retrieval-augmented question answering.",
        version="0.1.0",
    )

    embedding_model = SentenceTransformerEmbeddingModel()

    vector_store = QdrantVectorStore(
        collection_name="rag_documents",
        vector_size=384,
    )

    document_store = QdrantDocumentStore(
        collection_name="documents",
    )

    generator = GeminiGenerator()

    ingestion_service = IngestionService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        document_store=document_store,
    )

    rag_service = RAGService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        generator=generator,
    )

    app.state.ingestion_service = ingestion_service
    app.state.document_store = document_store
    app.state.rag_service = rag_service

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()