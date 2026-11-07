
from datetime import datetime, timezone
from uuid import uuid4

from app.document_store.postgres import PostgresDocumentStore
from app.models.document_record import DocumentRecord


def test_postgres_document_store_crud():

    store = PostgresDocumentStore()

    content_hash = uuid4().hex  # Unique test hash

    document = DocumentRecord(
        id=str(uuid4()),
        name="test_document.pdf",
        mime_type="application/pdf",
        content_hash=content_hash,
        total_chunks=5,
        status="processing",
        created_at=datetime.now(timezone.utc),
    )

    # 1. CREATE
    store.create(document)

    # 2. GET BY ID
    retrieved = store.get_by_id(document.id)

    assert retrieved is not None
    assert retrieved.name == "test_document.pdf"
    assert retrieved.status == "processing"

    # 3. GET BY HASH
    retrieved_by_hash = store.get_by_hash(content_hash)

    assert retrieved_by_hash is not None
    assert retrieved_by_hash.id == document.id

    # 4. LIST ALL
    documents = store.list_all()

    assert any(doc.id == document.id for doc in documents)

    # 5. UPDATE
    document.status = "completed"
    document.total_chunks = 10

    store.update(document)

    updated = store.get_by_id(document.id)

    assert updated.status == "completed"
    assert updated.total_chunks == 10

    # 6. DELETE
    store.delete(document.id)

    deleted = store.get_by_id(document.id)

    assert deleted is None

    print("All PostgreSQL DocumentStore CRUD operations passed!")
