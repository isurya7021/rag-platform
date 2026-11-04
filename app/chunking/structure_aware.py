from uuid import uuid4

from app.models.chunk import Chunk
from app.models.document import Document


class StructureAwareChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:

        chunks = []

        current_text = ""
        current_metadata = {}

        for element in document.elements:

            text = element.content.strip()

            if not text:
                continue

            # A heading starts a new section.
            if element.type == "heading":

                if current_text:
                    chunks.extend(
                        self._split_text(
                            current_text,
                            document,
                            current_metadata
                        )
                    )

                current_text = text
                current_metadata = element.metadata.copy()

                continue

            # Normal text belongs to current section.
            if current_text:
                current_text += "\n\n"

            current_text += text

            # Preserve metadata from the element.
            current_metadata.update(element.metadata)

        # Don't forget the final section.
        if current_text:
            chunks.extend(
                self._split_text(
                    current_text,
                    document,
                    current_metadata
                )
            )

        return chunks

    def _split_text(
        self,
        text: str,
        document: Document,
        metadata: dict
    ) -> list[Chunk]:

        chunks = []

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
                            **metadata
                        }
                    )
                )

            start += self.chunk_size - self.overlap

        return chunks