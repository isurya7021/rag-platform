
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
from threading import Event, Thread
from uuid import UUID
LEASE_RENEWAL_INTERVAL = 60  # seconds


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

        # 2. Prepare a proposed document record.
        mime_type, _ = mimetypes.guess_type(name)

        proposed_document = DocumentRecord(
            id=str(uuid4()),
            name=name,
            mime_type=mime_type or "application/octet-stream",
            content_hash=file_hash,
            total_chunks=0,
            status="processing",
            created_at=datetime.now(timezone.utc),
        )

        # 3. Atomically claim the document.
        claim = self.document_store.claim_document(
            proposed_document
        )

        document_record = claim.document

        # 4. Handle unsuccessful claims.
        if not claim.claimed:

            if document_record.status in ("completed", "empty"):
                return {
                    "document_id": document_record.id,
                    "filename": document_record.name,
                    "chunks_created": document_record.total_chunks,
                    "status": "already_exists",
                    "duplicate": True,
                }

            if document_record.status == "processing":
                return {
                    "document_id": document_record.id,
                    "filename": document_record.name,
                    "chunks_created": 0,
                    "status": "already_processing",
                    "duplicate": True,
                }

            raise RuntimeError(
                f"Document could not be claimed. "
                f"Current status: {document_record.status}"
            )

        claim_token = claim.claim_token
        generation_id = claim.generation_id

        if claim_token is None or generation_id is None:
            raise RuntimeError(
                "Document was claimed without a claim token or generation ID."
            )

        generation_id_str = str(generation_id)

        document_id = document_record.id

        stop_heartbeat, heartbeat_thread, ensure_lease = (
            self._start_lease_heartbeat(
                document_id,
                claim_token,
            )
        )

        try:
            ensure_lease()

            # 6. Parse the document.
            start = time.perf_counter()

            parser = ParserFactory.get_parser(str(path))
            document = parser.parse(str(path))
            ensure_lease()

            document.id = document_id
            document.name = name

            document_record.mime_type = document.mime_type
            self.document_store.update_claimed(
                document_record,
                claim_token,
            )

            print(
                f"[TIMING] Parsing: {time.perf_counter() - start:.2f}s",
                flush=True,
            )

            # 7. Generate chunks.
            start = time.perf_counter()

            chunks = self.chunker.chunk(document)
            ensure_lease()
            

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
                self.document_store.update_claimed(
                    document_record,
                    claim_token,
                )

                return {
                    "document_id": document_id,
                    "filename": name,
                    "chunks_created": 0,
                    "status": "empty",
                    "duplicate": False,
                }

            # 8. Attach document-level metadata to every chunk.
            for index, chunk in enumerate(chunks):
                chunk.metadata["document_id"] = document_id
                chunk.metadata["document_name"] = name
                chunk.metadata["document_hash"] = file_hash
                chunk.metadata["chunk_index"] = index
                chunk.metadata["generation_id"] = generation_id_str

            # 9. Generate embeddings.
            texts = [chunk.content for chunk in chunks]

            start = time.perf_counter()
            vectors = self.embedding_model.embed_batch(texts)
            ensure_lease()

            print(
                f"[TIMING] Embedding: {time.perf_counter() - start:.2f}s",
                flush=True,
            )

            if len(vectors) != len(chunks):
                raise ValueError(
                    "Number of embeddings does not match number of chunks"
                )

            # 10. Store chunks and vectors.
            start = time.perf_counter()

            ensure_lease()
            self.vector_store.upsert(
                chunks=chunks,
                vectors=vectors,
            )

            print(
                f"[TIMING] Qdrant upsert: {time.perf_counter() - start:.2f}s",
                flush=True,
            )

            ensure_lease()

            # 11. Mark document ingestion as completed.
            document_record.total_chunks = len(chunks)
            document_record.status = "completed"

            self.document_store.update_claimed(
                document_record,
                claim_token,
            )

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

            # 12. First verify ownership and prevent another retry
            # from starting while cleanup is in progress.
            try:
                document_record.status = "failed_cleanup"

                self.document_store.update_claimed(
                    document_record,
                    claim_token,
                )

            except Exception as ownership_error:
                print(
                    f"Could not transition document to failed_cleanup. "
                    f"Skipping vector cleanup because claim ownership "
                    f"could not be verified: {ownership_error}",
                    flush=True,
                )

                raise

            # 13. Clean up partially inserted vectors.
            try:
                self.vector_store.delete_by_document_id(document_id)

            except Exception as cleanup_error:
                print(
                    f"Failed to clean up document chunks: {cleanup_error}",
                    flush=True,
                )

            # 14. Mark the document as failed and allow a future retry.
            document_record.status = "failed"

            try:
                self.document_store.update_claimed(
                    document_record,
                    claim_token,
                )

            except Exception as update_error:
                print(
                    f"Could not update document failure status: {update_error}",
                    flush=True,
                )

            raise
        
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=5)

    def _start_lease_heartbeat(
        self,
        document_id: str,
        claim_token: UUID,
    ):
        stop_event = Event()
        heartbeat_errors = []

        def heartbeat():
            while not stop_event.wait(LEASE_RENEWAL_INTERVAL):
                try:
                    self.document_store.renew_lease(
                        document_id,
                        claim_token,
                    )

                except Exception as exc:
                    heartbeat_errors.append(exc)
                    stop_event.set()
                    return

        thread = Thread(
            target=heartbeat,
            daemon=True,
            name=f"lease-heartbeat-{document_id}",
        )

        thread.start()

        def ensure_lease():
            if heartbeat_errors:
                raise RuntimeError(
                    "Document lease renewal failed. "
                    "Worker may have lost ownership."
                ) from heartbeat_errors[0]

        return stop_event, thread, ensure_lease