
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import uuid

import pytest
from sqlalchemy import update

from app.database.connection import SessionLocal
from app.database.models import DocumentModel
from app.document_store.postgres import PostgresDocumentStore
from app.models.document_record import DocumentRecord


def create_document(content_hash):
    return DocumentRecord(
        id=str(uuid4()),
        name="publication_test.pdf",
        mime_type="application/pdf",
        content_hash=content_hash,
        total_chunks=0,
        status="processing",
        created_at=datetime.now(timezone.utc),
    )


def test_publish_generation_successfully():
    store = PostgresDocumentStore()
    document = create_document(uuid4().hex)

    try:
        claim = store.claim_document(document)

        assert claim.claimed
        assert claim.claim_token is not None
        assert claim.generation_id is not None

        store.publish_generation(
            document_id=claim.document.id,
            claim_token=claim.claim_token,
            generation_id=claim.generation_id,
            total_chunks=100,
        )

        with SessionLocal() as session:
            model = session.get(
                DocumentModel,
                uuid.UUID(claim.document.id),
            )

            assert model.status == "completed"
            assert model.total_chunks == 100
            assert model.published_generation_id == claim.generation_id

    finally:
        existing = store.get_by_hash(document.content_hash)
        if existing:
            store.delete(existing.id)


def test_stale_worker_cannot_publish_generation():
    store = PostgresDocumentStore()
    document = create_document(uuid4().hex)

    try:
        worker_a = store.claim_document(document)

        assert worker_a.claimed

        # Simulate expiration of Worker A's lease.
        with SessionLocal() as session:
            session.execute(
                update(DocumentModel)
                .where(
                    DocumentModel.id == uuid.UUID(worker_a.document.id)
                )
                .values(
                    lease_expires_at=(
                        datetime.now(timezone.utc)
                        - timedelta(minutes=1)
                    )
                )
            )
            session.commit()

        # Worker B takes over.
        worker_b = store.claim_document(document)

        assert worker_b.claimed
        assert worker_b.generation_id != worker_a.generation_id

        # Worker A must not publish its old generation.
        with pytest.raises(RuntimeError):
            store.publish_generation(
                document_id=worker_a.document.id,
                claim_token=worker_a.claim_token,
                generation_id=worker_a.generation_id,
                total_chunks=100,
            )

        # Worker B should still be able to publish.
        store.publish_generation(
            document_id=worker_b.document.id,
            claim_token=worker_b.claim_token,
            generation_id=worker_b.generation_id,
            total_chunks=100,
        )

        with SessionLocal() as session:
            model = session.get(
                DocumentModel,
                uuid.UUID(worker_b.document.id),
            )

            assert model.published_generation_id == worker_b.generation_id
            assert model.status == "completed"

    finally:
        existing = store.get_by_hash(document.content_hash)
        if existing:
            store.delete(existing.id)
