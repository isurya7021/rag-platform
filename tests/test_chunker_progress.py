
from app.chunking.sentence_aware import SentenceAwareChunker
from app.models.document import Document, DocumentElement

document = Document(
    id="test-document",
    name="test.txt",
    mime_type="text/plain",
    elements=[
        DocumentElement(
            type="text",
            content=(
                "A" * 480
                + ". "
                + "B" * 100
                + ". "
                + "C" * 100
                + "."
            ),
        )
    ],
)

chunker = SentenceAwareChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk(document)

print("Chunks created:", len(chunks))
for chunk in chunks:
    print("Chunk length:", len(chunk.content))
