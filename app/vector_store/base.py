from abc import ABC, abstractmethod

from app.models.chunk import Chunk


class VectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        generation_id: str,
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        published_generations: list[tuple[str, str]] | None = None,
    ) -> list[Chunk]:
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        pass

    @abstractmethod
    def delete_by_generation(
        self,
        document_id: str,
        generation_id: str,
    ) -> None:
        pass