from abc import ABC, abstractmethod

from app.models.document_record import DocumentRecord
from app.models.document_claim import DocumentClaim
from uuid import UUID


class DocumentStore(ABC):

    @abstractmethod
    def create(self, document: DocumentRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, document_id: str) -> DocumentRecord | None:
        pass

    @abstractmethod
    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        pass

    @abstractmethod
    def list_all(self) -> list[DocumentRecord]:
        pass

    @abstractmethod
    def update(self, document: DocumentRecord) -> None:
        pass

    @abstractmethod
    def delete(self, document_id: str) -> None:
        pass

    @abstractmethod
    def claim_document(
        self,
        document: DocumentRecord,
    ) -> DocumentClaim:
        pass

    @abstractmethod
    def update_claimed(
        self,
        document: DocumentRecord,
        claim_token: UUID,
    ) -> None:
        pass

    @abstractmethod
    def renew_lease(
        self,
        document_id: str,
        claim_token: UUID,
    ) -> None:
        pass