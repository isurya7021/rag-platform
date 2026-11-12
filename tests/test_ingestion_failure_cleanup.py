
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.ingestion.service import IngestionService
from app.models.chunk import Chunk
from app.models.document_claim import DocumentClaim
from uuid import uuid4


class FakeDocumentStore:
    def __init__(self):
        self.documents = {}

    def create(self, document):
        self.documents[document.id] = document

    def get_by_id(self, document_id):
        return self.documents.get(document_id)

    def get_by_hash(self, content_hash):
        return next(
            (
                doc for doc in self.documents.values()
                if doc.content_hash == content_hash
            ),
            None,
        )

    def list_all(self):
        return list(self.documents.values())

    def update(self, document):
        self.documents[document.id] = document

    def delete(self, document_id):
        self.documents.pop(document_id, None)

    def update_claimed(self, document, claim_token):
        self.update(document)

    def claim_document(self, document):
        existing = self.get_by_hash(document.content_hash)

        if existing is None:
            self.create(document)

            return DocumentClaim(
                document=document,
                claimed=True,
                claim_token=uuid4(),
                generation_id=uuid4(),
            )

        if existing.status == "failed":
            existing.status = "processing"
            self.update(existing)

            return DocumentClaim(
                document=existing,
                claimed=True,
                claim_token=uuid4(),
                generation_id=uuid4(),
            )

        return DocumentClaim(
            document=existing,
            claimed=False,
            claim_token=None,
            generation_id=None,
        )
    
    def renew_lease(self, document_id, claim_token):
        # Heartbeat is not triggered in the short-running unit tests.
        pass


class FakeVectorStore:
    def __init__(self):
        self.chunks = []
        self.fail_next_upsert = True

    def upsert(self, chunks, vectors):
        if self.fail_next_upsert:
            self.fail_next_upsert = False

            # Simulate partial insertion: only the first chunk is stored.
            self.chunks.extend(chunks[:1])

            raise RuntimeError("Simulated Qdrant upsert failure")

        self.chunks.extend(chunks)

    def delete_by_document_id(self, document_id):
        self.chunks = [
            chunk for chunk in self.chunks
            if chunk.metadata.get("document_id") != document_id
        ]

    def search(self, *args, **kwargs):
        return []


class FakeEmbeddingModel:
    def embed_batch(self, texts):
        return [[0.1] * 384 for _ in texts]


class FakeChunker:
    def chunk(self, document):
        return [
            Chunk(id="chunk-1", content="First chunk"),
            Chunk(id="chunk-2", content="Second chunk"),
            Chunk(id="chunk-3", content="Third chunk"),
        ]

class FakeGenerator:

    def generate(self, query, context):
        return "This is a mocked response."

def test_failed_ingestion_cleanup_and_retry():
    document_store = FakeDocumentStore()
    vector_store = FakeVectorStore()

    service = IngestionService(
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
        document_store=document_store,
        chunker=FakeChunker(),
    )

    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.txt"
        file_path.write_text("Test document content")

        # First attempt: simulate partial upsert failure.
        with pytest.raises(RuntimeError, match="Simulated Qdrant"):
            service.ingest(str(file_path))

        document = document_store.get_by_hash(
            service_hash(file_path)
        )

        assert document is not None
        assert document.status == "failed"
        assert len(vector_store.chunks) == 0

        # Second attempt: retry the same document.
        result = service.ingest(str(file_path))

        assert result["status"] == "ingested"
        assert result["chunks_created"] == 3

        assert document.status == "completed"
        assert len(vector_store.chunks) == 3

        print("Failure cleanup and retry test passed!")


def service_hash(file_path):
    from app.utils.file_hash import calculate_file_hash
    return calculate_file_hash(file_path)
