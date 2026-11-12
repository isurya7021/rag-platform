
from app.vector_store.qdrant import QdrantVectorStore
from app.models.chunk import Chunk


def test_generation_point_ids_are_different():
    document_id = "document-1"
    chunk_id = "chunk-1"

    point_g1 = QdrantVectorStore._generate_point_id(
        document_id,
        "generation-1",
        chunk_id,
    )

    point_g2 = QdrantVectorStore._generate_point_id(
        document_id,
        "generation-2",
        chunk_id,
    )

    assert point_g1 != point_g2


def test_same_generation_produces_same_point_id():
    point_1 = QdrantVectorStore._generate_point_id(
        "document-1",
        "generation-1",
        "chunk-1",
    )

    point_2 = QdrantVectorStore._generate_point_id(
        "document-1",
        "generation-1",
        "chunk-1",
    )

    assert point_1 == point_2
