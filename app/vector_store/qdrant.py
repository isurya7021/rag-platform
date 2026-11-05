from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.models.chunk import Chunk
from app.vector_store.base import VectorStore


class QdrantVectorStore(VectorStore):

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port
        )

        self._create_collection(vector_size)
        self.create_payload_index()

    def _create_collection(
        self,
        vector_size: int
    ) -> None:

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]]
    ) -> None:

        points = []

        for chunk, vector in zip(chunks, vectors):

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "content": chunk.content,
                        **chunk.metadata
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5
    ) -> list[Chunk]:

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        ).points

        return [
            Chunk(
                id=str(result.id),
                content=result.payload["content"],
                metadata={
                    key: value
                    for key, value in result.payload.items()
                    if key != "content"
                },
                score=result.score
            )
            for result in results
        ]
    
    def recreate_collection(self, vector_size: int) -> None:
        self.client.delete_collection(
            collection_name=self.collection_name
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    
    def find_document_id_by_hash(
        self,
        document_hash: str
    ) -> str | None:

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_hash",
                        match=MatchValue(value=document_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        return points[0].payload.get("document_id")
    

    def create_payload_index(self) -> None:

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_hash",
            field_schema="keyword",
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema="keyword",
        )


    def delete_by_document_id(self, document_id: str) -> None:

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )