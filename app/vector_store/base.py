from abc import ABC, abstractmethod

from app.models.chunk import Chunk


class VectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]]
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        limit: int = 5
    ) -> list[Chunk]:
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        pass