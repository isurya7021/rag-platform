
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import update as sql_update
from datetime import datetime, timezone


from app.database.connection import SessionLocal
from app.database.models import DocumentModel
from app.document_store.base import DocumentStore
from app.models.document_record import DocumentRecord
from app.models.document_claim import DocumentClaim


class PostgresDocumentStore(DocumentStore):

    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    
    @staticmethod
    def _to_record(model: DocumentModel) -> DocumentRecord:
        return DocumentRecord(
            id=str(model.id),
            name=model.name,
            mime_type=model.mime_type,
            content_hash=model.content_hash,
            total_chunks=model.total_chunks,
            status=model.status,
            created_at=model.created_at,
        )

    @staticmethod
    def _to_model(document: DocumentRecord) -> DocumentModel:
        return DocumentModel(
            id=UUID(document.id),
            name=document.name,
            mime_type=document.mime_type,
            content_hash=document.content_hash,
            total_chunks=document.total_chunks,
            status=document.status,
            created_at=document.created_at,
        )

    def create(self, document: DocumentRecord) -> None:
        with self.session_factory() as session:
            model = self._to_model(document)
            session.add(model)
            session.commit()

    def get_by_id(self, document_id: str) -> DocumentRecord | None:
        with self.session_factory() as session:
            model = session.get(DocumentModel, UUID(document_id))

            if model is None:
                return None

            return self._to_record(model)

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self.session_factory() as session:
            statement = select(DocumentModel).where(
                DocumentModel.content_hash == content_hash
            )

            model = session.scalar(statement)

            if model is None:
                return None

            return self._to_record(model)
        
    def list_all(self) -> list[DocumentRecord]:
        with self.session_factory() as session:
            statement = select(DocumentModel).order_by(
                DocumentModel.created_at.desc()
            )

            models = session.scalars(statement).all()

            return [
                self._to_record(model)
                for model in models
            ]
        
    def update(self, document: DocumentRecord) -> None:
        with self.session_factory() as session:
            model = session.get(DocumentModel, UUID(document.id))

            if model is None:
                raise ValueError(
                    f"Document not found: {document.id}"
                )

            model.name = document.name
            model.mime_type = document.mime_type
            model.content_hash = document.content_hash
            model.total_chunks = document.total_chunks
            model.status = document.status
            session.commit()

    def delete(self, document_id: str) -> None:
        with self.session_factory() as session:
            model = session.get(DocumentModel, UUID(document_id))

            if model is not None:
                session.delete(model)
                session.commit()

    
    
    def claim_document(
        self,
        document: DocumentRecord,
    ) -> DocumentClaim:

        with self.session_factory() as session:

            # 1. Try to insert a new document.
            statement = (
                insert(DocumentModel)
                .values(
                    id=UUID(document.id),
                    name=document.name,
                    mime_type=document.mime_type,
                    content_hash=document.content_hash,
                    total_chunks=0,
                    status="processing",
                    created_at=document.created_at,
                )
                .on_conflict_do_nothing(
                    index_elements=["content_hash"]
                )
                .returning(DocumentModel.id)
            )

            inserted_id = session.execute(statement).scalar_one_or_none()

            # 2. New document successfully claimed.
            if inserted_id is not None:
                session.commit()

                return DocumentClaim(
                    document=document,
                    claimed=True,
                )

            # 3. Document already exists. Try to claim it
            #    only if its previous status is failed.
            retry_statement = (
                sql_update(DocumentModel)
                .where(
                    DocumentModel.content_hash == document.content_hash,
                    DocumentModel.status == "failed",
                )
                .values(
                    status="processing",
                    total_chunks=0,
                    updated_at=datetime.now(timezone.utc),
                )
                .returning(DocumentModel)
            )

            existing_model = session.scalars(
                retry_statement
            ).one_or_none()

            if existing_model is not None:
                record = self._to_record(existing_model)

                session.commit()

                return DocumentClaim(
                    document=record,
                    claimed=True,
                )

            # 4. Another request owns it, or it is already completed.
            existing_model = session.scalar(
                select(DocumentModel).where(
                    DocumentModel.content_hash == document.content_hash
                )
            )

            if existing_model is None:
                raise RuntimeError(
                    "Document disappeared during claim operation"
                )

            record = self._to_record(existing_model)

            session.commit()

            return DocumentClaim(
                document=record,
                claimed=False,
            )

