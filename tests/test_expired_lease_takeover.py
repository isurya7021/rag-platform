
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import update

from app.database.connection import SessionLocal
from app.database.models import DocumentModel
from app.document_store.postgres import PostgresDocumentStore
from app.models.document_record import DocumentRecord


def test_concurrent_expired_lease_takeover():
    store = PostgresDocumentStore()
    content_hash = uuid4().hex

    initial_document = DocumentRecord(
        id=str(uuid4()),
        name="expired_lease_test.pdf",
        mime_type="application/pdf",
        content_hash=content_hash,
        total_chunks=0,
        status="processing",
        created_at=datetime.now(timezone.utc),
    )

    try:
        # 1. Worker A claims the document.
        initial_claim = store.claim_document(initial_document)

        assert initial_claim.claimed is True
        assert initial_claim.claim_token is not None

        initial_generation_id = initial_claim.generation_id
        assert initial_generation_id is not None

        document_id = initial_claim.document.id

        # 2. Simulate Worker A crashing by expiring its lease.
        with SessionLocal() as session:
            session.execute(
                update(DocumentModel)
                .where(DocumentModel.id == document_id)
                .values(
                    lease_expires_at=datetime.now(timezone.utc)
                    - timedelta(minutes=1)
                )
            )
            session.commit()

        # 3. Ten workers simultaneously attempt to take over.
        def attempt_takeover(_):
            proposed_document = DocumentRecord(
                id=str(uuid4()),
                name="expired_lease_test.pdf",
                mime_type="application/pdf",
                content_hash=content_hash,
                total_chunks=0,
                status="processing",
                created_at=datetime.now(timezone.utc),
            )

            return store.claim_document(proposed_document)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(attempt_takeover, range(10)))

        successful_claims = [
            result for result in results if result.claimed
        ]
        unsuccessful_claims = [
            result for result in results if not result.claimed
        ]

        # 4. Exactly one worker must acquire the expired lease.
        assert len(successful_claims) == 1
        assert len(unsuccessful_claims) == 9

        winning_claim = successful_claims[0]

        assert winning_claim.generation_id is not None

        assert winning_claim.generation_id != initial_generation_id

        assert all(
            result.generation_id is None
            for result in unsuccessful_claims
        )

        # 5. All workers must refer to the same document.
        document_ids = {result.document.id for result in results}
        assert document_ids == {document_id}

        # 6. Verify that the database contains the winning token.
        with SessionLocal() as session:
            stored_document = session.get(DocumentModel, document_id)

            assert stored_document is not None
            assert stored_document.status == "processing"
            assert stored_document.claim_token == winning_claim.claim_token

        print("Initial claim: successful")
        print("Expired lease takeover: successful")
        print(f"Successful takeover attempts: {len(successful_claims)}")
        print(f"Rejected takeover attempts: {len(unsuccessful_claims)}")

    finally:
        stored_document = store.get_by_hash(content_hash)

        if stored_document is not None:
            store.delete(stored_document.id)
