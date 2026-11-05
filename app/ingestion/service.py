from pathlib import Path
import mimetypes
import time
from datetime import datetime, timezone
from uuid import uuid4

from app.parsers.factory import ParserFactory
from app.chunking.sentence_aware import SentenceAwareChunker
from app.embeddings.base import EmbeddingModel
from app.vector_store.base import VectorStore
from app.document_store.base import DocumentStore
from app.models.document_record import DocumentRecord
from app.utils.file_hash import calculate_file_hash


class IngestionService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        document_store: DocumentStore,
        chunker: SentenceAwareChunker | None = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.document_store = document_store
        self.chunker = chunker or SentenceAwareChunker()

    def ingest(
        self,
        file_path: str,
        document_name: str | None = None,
    ) -> dict:

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        name = document_name or path.name

        # 1. Calculate the document's SHA-256 hash.
        file_hash = calculate_file_hash(path)

        # 2. Check the document registry.
        existing_document = self.document_store.get_by_hash(file_hash)

        if existing_document:

            if existing_document.status in ("completed", "empty"):
                print(f"Document already exists: {name}")

                return {
                    "document_id": existing_document.id,
                    "filename": existing_document.name,
                    "chunks_created": existing_document.total_chunks,
                    "status": "already_exists",
                    "duplicate": True,
                }

            if existing_document.status == "processing":
                return {
                    "document_id": existing_document.id,
                    "filename": existing_document.name,
                    "chunks_created": 0,
                    "status": "already_processing",
                    "duplicate": True,
                }

            # A previous attempt failed. Reuse its document ID.
            document_record = existing_document

            # Remove chunks left behind by the previous attempt.
            self.vector_store.delete_by_document_id(document_record.id)

            document_record.status = "processing"
            document_record.total_chunks = 0
            self.document_store.update(document_record)

        else:

            # 3. Create a new document record.
            document_id = str(uuid4())

            mime_type, _ = mimetypes.guess_type(name)

            document_record = DocumentRecord(
                id=document_id,
                name=name,
                mime_type=mime_type or "application/octet-stream",
                content_hash=file_hash,
                total_chunks=0,
                status="processing",
                created_at=datetime.now(timezone.utc),
            )

            self.document_store.create(document_record)

        document_id = document_record.id

        try:
            # 4. Parse the document.
            start = time.perf_counter()

            parser = ParserFactory.get_parser(str(path))
            document = parser.parse(str(path))

            document.id = document_id
            document.name = name

            document_record.mime_type = document.mime_type
            self.document_store.update(document_record)

            print(
                f"[TIMING] Parsing: {time.perf_counter() - start:.2f}s",
                flush=True,
            )

            # 5. Generate chunks.
            start = time.perf_counter()

            chunks = self.chunker.chunk(document)

            print(
                f"[TIMING] Chunking: {time.perf_counter() - start:.2f}s",
                flush=True,
            )

            print(
                f"[TIMING] Chunks produced: {len(chunks)}",
                flush=True,
            )

            if not chunks:
                document_record.status = "empty"
                document_record.total_chunks = 0
                self.document_store.update(document_record)

                return {
                    "document_id": document_id,
                    "filename": name,
                    "chunks_created": 0,
                    "status": "empty",
                    "duplicate": False,
                }

            # 6. Attach document-level metadata to every chunk.
            for index, chunk in enumerate(chunks):
                chunk.metadata["document_id"] = document_id
                chunk.metadata["document_name"] = name
                chunk.metadata["document_hash"] = file_hash
                chunk.metadata["chunk_index"] = index

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

            # 9. Mark document ingestion as completed.
            document_record.total_chunks = len(chunks)
            document_record.status = "completed"

            self.document_store.update(document_record)

            print(f"Document: {name}")
            print(f"Document ID: {document_id}")
            print(f"Document hash: {file_hash}")
            print(f"Chunks created: {len(chunks)}")

            return {
                "document_id": document_id,
                "filename": name,
                "chunks_created": len(chunks),
                "status": "ingested",
                "duplicate": False,
            }

        except Exception:
            # Try to remove partially inserted chunks.
            try:
                self.vector_store.delete_by_document_id(document_record.id)
            except Exception as cleanup_error:
                print(
                    f"Failed to clean up document chunks: {cleanup_error}",
                    flush=True,
                )

            # Mark the document as failed.
            document_record.status = "failed"

            try:
                self.document_store.update(document_record)
            except Exception as update_error:
                print(
                    f"Could not update document failure status: {update_error}",
                    flush=True,
                )

            raise