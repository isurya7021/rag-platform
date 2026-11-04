from uuid import uuid4

from app.models.chunk import Chunk
from app.models.document import Document


class FixedSizeChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:

        chunks = []

        for element in document.elements:

            text = element.content

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        Chunk(
                            id=str(uuid4()),
                            content=chunk_text,
                            metadata={
                                "document_id": document.id,
                                "document_name": document.name,
                                **element.metadata
                            }
                        )
                    )

                start += self.chunk_size - self.overlap

        return chunks