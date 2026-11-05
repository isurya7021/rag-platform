from datetime import datetime, timezone
from uuid import uuid4

from app.document_store.qdrant import QdrantDocumentStore
from app.models.document_record import DocumentRecord


def test_document_crud():

    store = QdrantDocumentStore(
        collection_name="documents",
    )

    # 1. Create a document record.
    document = DocumentRecord(
        id=str(uuid4()),
        name="sample.pdf",
        mime_type="application/pdf",
        content_hash="test_hash_123",
        total_chunks=9,
        status="completed",
        created_at=datetime.now(timezone.utc),
    )

    store.create(document)

    print("\n1. Document created")

    # 2. Retrieve document by ID.
    retrieved = store.get_by_id(document.id)

    assert retrieved is not None
    assert retrieved.name == "sample.pdf"

    print("2. Retrieved by ID:", retrieved.name)

    # 3. Retrieve document by content hash.
    retrieved_by_hash = store.get_by_hash("test_hash_123")

    assert retrieved_by_hash is not None
    assert retrieved_by_hash.id == document.id

    print("3. Retrieved by hash:", retrieved_by_hash.id)

    # 4. List all documents.
    documents = store.list_all()

    assert any(
        item.id == document.id
        for item in documents
    )

    print("4. Total documents:", len(documents))

    # 5. Update document status.
    document.status = "processing"

    store.update(document)

    updated = store.get_by_id(document.id)

    assert updated is not None
    assert updated.status == "processing"

    print("5. Updated status:", updated.status)

    # 6. Delete document.
    store.delete(document.id)

    deleted = store.get_by_id(document.id)

    assert deleted is None

    print("6. Document deleted successfully")


if __name__ == "__main__":
    test_document_crud() 