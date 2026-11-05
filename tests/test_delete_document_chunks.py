from app.vector_store.qdrant import QdrantVectorStore


def test_delete_document_chunks():

    store = QdrantVectorStore(
        collection_name="rag_documents",
        vector_size=384,
    )

    document_id = "f5df1fa1-38e9-4f4e-8d36-8dc47008179e"

    # Delete chunks belonging to this document.
    store.delete_by_document_id(document_id)

    # Verify that no chunks remain.
    points, _ = store.client.scroll(
        collection_name="rag_documents",
        scroll_filter={
            "must": [
                {
                    "key": "document_id",
                    "match": {"value": document_id},
                }
            ]
        },
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    assert len(points) == 0

    print("All chunks for the document were deleted!")


if __name__ == "__main__":
    test_delete_document_chunks()