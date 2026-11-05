from abc import ABC, abstractmethod

from app.models.document_record import DocumentRecord


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