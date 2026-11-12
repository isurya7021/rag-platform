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
from uuid import uuid5, NAMESPACE_URL


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
        vectors: list[list[float]],
        generation_id: str,
    ) -> None:

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks and vectors must match"
            )

        points = []

        for chunk, vector in zip(chunks, vectors):

            document_id = chunk.metadata["document_id"]

            point_id = self._generate_point_id(
                document_id=document_id,
                generation_id=generation_id,
                chunk_id=chunk.id,
            )

            payload = {
                "content": chunk.content,
                **chunk.metadata,
                "generation_id": generation_id,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        published_generations: list[tuple[str, str]] | None = None,
    ) -> list[Chunk]:

        # No published generations means no searchable documents.
        if not published_generations:
            return []

        # Build one condition for each document-generation pair.
        generation_filters = [
            Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                    FieldCondition(
                        key="generation_id",
                        match=MatchValue(value=generation_id),
                    ),
                ]
            )
            for document_id, generation_id in published_generations
        ]

        # Search only within the published generations.
        search_filter = Filter(
            should=generation_filters
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
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
                score=result.score,
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

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="generation_id",
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

    @staticmethod
    def _generate_point_id(
        document_id: str,
        generation_id: str,
        chunk_id: str,
    ) -> str:
        unique_key = f"{document_id}:{generation_id}:{chunk_id}"

        return str(uuid5(NAMESPACE_URL, unique_key))
    
    def delete_by_generation(
        self,
        document_id: str,
        generation_id: str,
    ) -> None:

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                    FieldCondition(
                        key="generation_id",
                        match=MatchValue(value=generation_id),
                    ),
                ]
            ),
            wait=True,
        )

    