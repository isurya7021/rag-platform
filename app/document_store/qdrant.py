from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.models.document_record import DocumentRecord
from app.document_store.base import DocumentStore


class QdrantDocumentStore(DocumentStore):

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port,
        )

        self._create_collection()

    def _create_collection(self) -> None:

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={},
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="content_hash",
            field_schema="keyword",
        )

    def _to_payload(self, document: DocumentRecord) -> dict:

        return {
            "name": document.name,
            "mime_type": document.mime_type,
            "content_hash": document.content_hash,
            "total_chunks": document.total_chunks,
            "status": document.status,
            "created_at": document.created_at.isoformat(),
        }

    def _to_document(self, point) -> DocumentRecord:

        payload = point.payload

        return DocumentRecord(
            id=str(point.id),
            name=payload["name"],
            mime_type=payload["mime_type"],
            content_hash=payload["content_hash"],
            total_chunks=payload["total_chunks"],
            status=payload["status"],
            created_at=datetime.fromisoformat(
                payload["created_at"]
            ),
        )

    def create(self, document: DocumentRecord) -> None:

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=document.id,
                    vector={},
                    payload=self._to_payload(document),
                )
            ],
        )

    def get_by_id(
        self,
        document_id: str,
    ) -> DocumentRecord | None:

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[document_id],
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        return self._to_document(points[0])

    def get_by_hash(
        self,
        content_hash: str,
    ) -> DocumentRecord | None:

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_hash",
                        match=MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        return self._to_document(points[0])

    def list_all(self) -> list[DocumentRecord]:

        documents = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            documents.extend(
                self._to_document(point)
                for point in points
            )

            if next_offset is None:
                break

            offset = next_offset

        return documents

    def update(self, document: DocumentRecord) -> None:

        self.client.set_payload(
            collection_name=self.collection_name,
            payload=self._to_payload(document),
            points=[document.id],
        )

    def delete(self, document_id: str) -> None:

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[document_id],
        )