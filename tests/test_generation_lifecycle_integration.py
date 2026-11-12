
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import update

from app.database.connection import SessionLocal
from app.database.models import DocumentModel
from app.document_store.postgres import PostgresDocumentStore
from app.models.chunk import Chunk
from app.models.document_record import DocumentRecord
from app.rag.service import RAGService
from app.vector_store.qdrant import QdrantVectorStore


VECTOR_SIZE = 384
COLLECTION_NAME = f"rag_generation_test_{uuid4().hex}"

class FakeEmbeddingModel:
    """Returns a deterministic vector for integration testing."""

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (VECTOR_SIZE - 1)


class FakeGenerator:
    """Avoids making real Gemini API calls."""

    def generate(self, question: str, context: str) -> str:
        return f"Test answer based on: {context}"


def create_document(name: str) -> DocumentRecord:
    return DocumentRecord(
        id=str(uuid4()),
        name=name,
        mime_type="application/pdf",
        content_hash=uuid4().hex,
        total_chunks=0,
        status="processing",
        created_at=datetime.now(timezone.utc),
    )


def create_chunk(document: DocumentRecord, content: str) -> Chunk:
    return Chunk(
        id=str(uuid4()),
        content=content,
        metadata={
            "document_id": document.id,
            "document_name": document.name,
            "document_hash": document.content_hash,
        },
    )


def upsert_generation(
    vector_store: QdrantVectorStore,
    document: DocumentRecord,
    generation_id,
    content: str,
) -> None:
    chunk = create_chunk(document, content)

    vector_store.upsert(
        chunks=[chunk],
        vectors=[[1.0] + [0.0] * (VECTOR_SIZE - 1)],
        generation_id=str(generation_id),
    )


def mark_document_failed(document_id: str) -> None:
    with SessionLocal() as session:
        session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(status="failed")
        )
        session.commit()


def get_sources(rag_service: RAGService) -> set[str]:
    response = rag_service.query(
        question="Explain the test document",
        top_k=10,
    )

    return {
        source["content"]
        for source in response["sources"]
    }


def test_generation_lifecycle_with_real_stores():

    document_store = PostgresDocumentStore()

    vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_SIZE,
    )

    rag_service = RAGService(
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
        document_store=document_store,
        generator=FakeGenerator(),
    )

    document = create_document("generation_lifecycle_test.pdf")

    other_document = create_document("other_generation_test.pdf")

    document_ids = [
        document.id,
        other_document.id,
    ]

    try:
        # --------------------------------------------------
        # 1. Initial ingestion: publish G1
        # --------------------------------------------------

        claim_g1 = document_store.claim_document(document)

        assert claim_g1.claimed
        assert claim_g1.generation_id is not None

        upsert_generation(
            vector_store,
            claim_g1.document,
            claim_g1.generation_id,
            "Content from generation G1",
        )

        document_store.publish_generation(
            document_id=claim_g1.document.id,
            claim_token=claim_g1.claim_token,
            generation_id=claim_g1.generation_id,
            total_chunks=1,
        )

        # G1 should be searchable.
        assert get_sources(rag_service) == {
            "Content from generation G1"
        }

        # --------------------------------------------------
        # 2. Start another ingestion: G2
        # --------------------------------------------------

        mark_document_failed(claim_g1.document.id)

        claim_g2 = document_store.claim_document(document)

        assert claim_g2.claimed
        assert claim_g2.generation_id != claim_g1.generation_id

        upsert_generation(
            vector_store,
            claim_g2.document,
            claim_g2.generation_id,
            "Content from generation G2",
        )

        # G2 exists in Qdrant but has not been published.
        # G1 must remain searchable.
        assert get_sources(rag_service) == {
            "Content from generation G1"
        }

        # --------------------------------------------------
        # 3. G2 fails
        # --------------------------------------------------

        mark_document_failed(claim_g2.document.id)

        # G1 must still be searchable after G2 fails.
        assert get_sources(rag_service) == {
            "Content from generation G1"
        }

        # --------------------------------------------------
        # 4. Retry and successfully publish G3
        # --------------------------------------------------

        claim_g3 = document_store.claim_document(document)

        assert claim_g3.claimed
        assert claim_g3.generation_id not in {
            claim_g1.generation_id,
            claim_g2.generation_id,
        }

        upsert_generation(
            vector_store,
            claim_g3.document,
            claim_g3.generation_id,
            "Content from generation G3",
        )

        document_store.publish_generation(
            document_id=claim_g3.document.id,
            claim_token=claim_g3.claim_token,
            generation_id=claim_g3.generation_id,
            total_chunks=1,
        )

        # G3 should replace G1 in retrieval.
        assert get_sources(rag_service) == {
            "Content from generation G3"
        }

        # --------------------------------------------------
        # 5. Verify isolation between documents
        # --------------------------------------------------

        other_claim = document_store.claim_document(other_document)

        assert other_claim.claimed

        upsert_generation(
            vector_store,
            other_claim.document,
            other_claim.generation_id,
            "Published content from another document",
        )

        document_store.publish_generation(
            document_id=other_claim.document.id,
            claim_token=other_claim.claim_token,
            generation_id=other_claim.generation_id,
            total_chunks=1,
        )

        # Add an unpublished generation for the second document.
        unpublished_generation = uuid4()

        upsert_generation(
            vector_store,
            other_claim.document,
            unpublished_generation,
            "Unpublished content from another document",
        )

        sources = get_sources(rag_service)

        assert sources == {
            "Content from generation G3",
            "Published content from another document",
        }

        assert "Content from generation G1" not in sources
        assert "Content from generation G2" not in sources
        assert "Unpublished content from another document" not in sources

    finally:
        # Delete only the test document records.
        for document_id in document_ids:
            document_store.delete(document_id)

        # Delete the isolated test collection.
        vector_store.client.delete_collection(
            collection_name=COLLECTION_NAME
        )
