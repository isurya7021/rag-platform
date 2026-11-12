
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from app.document_store.postgres import PostgresDocumentStore
from app.models.document_record import DocumentRecord


def test_concurrent_document_claim():

    store = PostgresDocumentStore()

    # Same hash for every concurrent request.
    content_hash = uuid4().hex

    def attempt_claim(_):
        document = DocumentRecord(
            id=str(uuid4()),
            name="concurrent_test.pdf",
            mime_type="application/pdf",
            content_hash=content_hash,
            total_chunks=0,
            status="processing",
            created_at=datetime.now(timezone.utc),
        )

        return store.claim_document(document)

    try:
        # Simulate 10 requests arriving concurrently.
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(attempt_claim, range(10)))

        successful_claims = [
            result for result in results if result.claimed
        ]

        unsuccessful_claims = [
            result for result in results if not result.claimed
        ]

        print(f"Successful claims: {len(successful_claims)}")
        print(f"Unsuccessful claims: {len(unsuccessful_claims)}")

        assert successful_claims[0].generation_id is not None
        assert all(
            result.generation_id is None
            for result in unsuccessful_claims
        )

        assert len(successful_claims) == 1
        assert len(unsuccessful_claims) == 9

        # Every request must refer to the same stored document.
        document_ids = {result.document.id for result in results}

        assert len(document_ids) == 1

        # Only one database record should exist.
        stored_document = store.get_by_hash(content_hash)

        assert stored_document is not None
        assert stored_document.status == "processing"

    finally:
        stored_document = store.get_by_hash(content_hash)

        if stored_document is not None:
            store.delete(stored_document.id)
