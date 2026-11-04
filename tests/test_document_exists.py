from app.vector_store.qdrant import QdrantVectorStore


def test_document_exists():
    vector_store = QdrantVectorStore(
        collection_name="rag_documents",
        vector_size=384,
    )

    test_hash = "test_sha256_hash_123"

    # Initially, this hash should not exist.
    result = vector_store.document_exists(test_hash)

    print("Document exists:", result)


if __name__ == "__main__":
    test_document_exists()