from pathlib import Path
import time
from uuid import uuid4

from app.parsers.factory import ParserFactory
from app.chunking.sentence_aware import SentenceAwareChunker
from app.embeddings.base import EmbeddingModel
from app.vector_store.base import VectorStore
from app.utils.file_hash import calculate_file_hash


class IngestionService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: SentenceAwareChunker | None = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunker = chunker or SentenceAwareChunker()

    # def ingest(
    #     self,
    #     file_path: str,
    #     document_name: str | None = None,
    # ) -> int:
    #     path = Path(file_path)

    #     if not path.is_file():
    #         raise FileNotFoundError(
    #             f"Document not found: {file_path}"
    #         )

    #     # 1. Parse the file into our common Document model.
    #     parser = ParserFactory.get_parser(str(path))
    #     document = parser.parse(str(path))
       
    #     if document_name:
    #         document.name = document_name

    #     # 2. Split the document into chunks.
    #     chunks = self.chunker.chunk(document)

    #     if not chunks:
    #         print(f"No text chunks found in {path.name}")
    #         return 0

    #     # 3. Generate one embedding per chunk.
    #     texts = [chunk.content for chunk in chunks]
    #     vectors = self.embedding_model.embed_batch(texts)

    #     if len(vectors) != len(chunks):
    #         raise ValueError(
    #             "Number of embeddings does not match number of chunks"
    #         )

    #     # 4. Store the chunks and their vectors.
    #     self.vector_store.upsert(
    #         chunks=chunks,
    #         vectors=vectors,
    #     )

    #     print(f"Document: {document.name}")
    #     print(f"Chunks created: {len(chunks)}")
    #     print(f"Vectors generated: {len(vectors)}")

    #     return len(chunks)
    
    # def ingest(self, file_path: str, document_name: str | None = None) -> int:
    #     path = Path(file_path)

    #     if not path.is_file():
    #         raise FileNotFoundError(f"Document not found: {file_path}")

    #     start = time.perf_counter()
    #     parser = ParserFactory.get_parser(str(path))
    #     document = parser.parse(str(path))

    #     if document_name:
    #         document.name = document_name

    #     print(
    #         f"[TIMING] Parsing: {time.perf_counter() - start:.2f}s",
    #         flush=True,
    #     )

    #     start = time.perf_counter()
    #     chunks = self.chunker.chunk(document)

    #     print(
    #         f"[TIMING] Chunking: {time.perf_counter() - start:.2f}s",
    #         flush=True,
    #     )
    #     print(f"[TIMING] Chunks produced: {len(chunks)}", flush=True)

    #     if not chunks:
    #         return 0

    #     texts = [chunk.content for chunk in chunks]

    #     start = time.perf_counter()
    #     vectors = self.embedding_model.embed_batch(texts)

    #     print(
    #         f"[TIMING] Embedding: {time.perf_counter() - start:.2f}s",
    #         flush=True,
    #     )

    #     if len(vectors) != len(chunks):
    #         raise ValueError("Number of embeddings does not match number of chunks")

    #     start = time.perf_counter()
    #     self.vector_store.upsert(chunks=chunks, vectors=vectors)

    #     print(
    #         f"[TIMING] Qdrant upsert: {time.perf_counter() - start:.2f}s",
    #         flush=True,
    #     )

    #     print(f"[TIMING] Document: {document.name}", flush=True)
    #     return len(chunks)
    

    def ingest(
        self,
        file_path: str,
        document_name: str | None = None,
    ) -> int:

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        # 1. Determine document name.
        name = document_name or path.name

        # 2. Calculate SHA-256 incrementally.
        file_hash = calculate_file_hash(path)

        # 3. Check whether the document already exists.
        if self.vector_store.document_exists(file_hash):
            print(f"Document already ingested: {name}")
            return 0

        # 4. Generate a new document ID.
        document_id = str(uuid4())

        # 5. Parse the document.
        parser = ParserFactory.get_parser(str(path))
        document = parser.parse(str(path))

        document.id = document_id
        document.name = name

        # 5. Generate chunks.
        chunks = self.chunker.chunk(document)

        if not chunks:
            print(f"No text chunks found in {name}")
            return 0

        # 6. Attach document metadata.
        for chunk in chunks:
            chunk.metadata["document_hash"] = file_hash

        # 7. Generate embeddings.
        texts = [chunk.content for chunk in chunks]

        start = time.perf_counter()
        vectors = self.embedding_model.embed_batch(texts)

        print(
            f"[TIMING] Embedding: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if len(vectors) != len(chunks):
            raise ValueError(
                "Number of embeddings does not match number of chunks"
            )

        # 8. Store chunks and vectors.
        start = time.perf_counter()

        self.vector_store.upsert(
            chunks=chunks,
            vectors=vectors,
        )

        print(
            f"[TIMING] Qdrant upsert: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        print(f"Document: {name}")
        print(f"Document ID: {document_id}")
        print(f"Document hash: {file_hash}")
        print(f"Chunks created: {len(chunks)}")

        return len(chunks)