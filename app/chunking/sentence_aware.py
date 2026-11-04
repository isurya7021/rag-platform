import re
from uuid import uuid4

from app.models.chunk import Chunk
from app.models.document import Document


class SentenceAwareChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        chunks = []

        current_section = ""
        current_metadata = {}
        current_heading = None

        for element in document.elements:

            text = element.content.strip()

            if not text:
                continue

            # A heading starts a new section.
            if element.type == "heading":

                if current_section:
                    chunks.extend(
                        self._create_chunks(
                            current_section,
                            document,
                            current_metadata
                        )
                    )

                current_heading = text

                current_section = ""

                current_metadata = {
                    "section": text,
                    **element.metadata
                }

            else:

                if current_section:
                    current_section += "\n\n"

                current_section += text

        # Process final section.
        if current_section:
            chunks.extend(
                self._create_chunks(
                    current_section,
                    document,
                    current_metadata
                )
            )

        return chunks

    
    def _create_chunks(
        self,
        text: str,
        document: Document,
        metadata: dict,
    ) -> list[Chunk]:

        sentences = self._split_sentences(text)
        chunks = []

        current_sentences = []
        current_length = 0
        i = 0

        while i < len(sentences):
            sentence = sentences[i]

            # Add the sentence if it fits, or if the chunk is empty.
            if (
                not current_sentences
                or current_length + len(sentence) <= self.chunk_size
            ):
                current_sentences.append(sentence)
                current_length += len(sentence)
                i += 1
                continue

            # Save the current chunk.
            chunks.append(
                self._create_chunk(
                    current_sentences,
                    document,
                    metadata,
                )
            )

            # Retain overlap, but guarantee forward progress.
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_sentences):
                if overlap_length + len(previous_sentence) > self.overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence)

            # If overlap prevents the next sentence from fitting,
            # discard the overlap and continue with the next sentence.
            if overlap_sentences and (
                overlap_length + len(sentence) > self.chunk_size
            ):
                overlap_sentences = []
                overlap_length = 0

            current_sentences = overlap_sentences
            current_length = overlap_length

        # Save the final chunk.
        if current_sentences:
            chunks.append(
                self._create_chunk(
                    current_sentences,
                    document,
                    metadata,
                )
            )

        return chunks


    def _split_sentences(self, text: str) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _create_chunk(
        self,
        sentences: list[str],
        document: Document,
        metadata: dict
    ) -> Chunk:

        content = " ".join(sentences)

        section = metadata.get("section")

        if section:
            content = f"Section: {section}\n\n{content}"

        return Chunk(
            id=str(uuid4()),
            content=content,
            metadata={
                "document_id": document.id,
                "document_name": document.name,
                **metadata
            }
        )