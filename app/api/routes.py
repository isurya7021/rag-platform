
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from uuid import UUID
import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field
from google.genai import errors

from app.ingestion.service import IngestionService
from app.rag.service import RAGService
from app.document_store.base import DocumentStore
from app.vector_store.base import VectorStore


logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html"}
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


# --------------------------------------------------
# Dependencies
# --------------------------------------------------

def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


def get_document_store(request: Request) -> DocumentStore:
    return request.app.state.document_store


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


# --------------------------------------------------
# Document ingestion
# --------------------------------------------------

@router.post("/documents")
async def ingest_document(
    file: Annotated[UploadFile, File()],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload TXT, PDF, DOCX, or HTML.",
        )

    temp_path = None

    try:
        with NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix="upload_",
        ) as temp_file:

            temp_path = Path(temp_file.name)

            while contents := await file.read(UPLOAD_CHUNK_SIZE):
                temp_file.write(contents)

        result = service.ingest(
            str(temp_path),
            document_name=filename,
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check server logs for details.",
        ) from exc

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# --------------------------------------------------
# Document query
# --------------------------------------------------

@router.post("/query")
def query_documents(
    request_body: QueryRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
):
    try:
        return service.query(
            question=request_body.question,
            top_k=request_body.top_k,
        )

    except errors.APIError as exc:
        logger.exception("LLM provider error")

        status_code = exc.code if exc.code and 400 <= exc.code <= 599 else 502

        raise HTTPException(
            status_code=status_code,
            detail="LLM provider request failed.",
        ) from exc

    except Exception as exc:
        logger.exception("Document query failed")

        raise HTTPException(
            status_code=500,
            detail="Query failed. Check server logs for details.",
        ) from exc


# --------------------------------------------------
# List documents
# --------------------------------------------------

@router.get("/documents")
def list_documents(
    store: Annotated[DocumentStore, Depends(get_document_store)],
):
    try:
        return store.list_all()

    except Exception as exc:
        logger.exception("Failed to retrieve documents")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents.",
        ) from exc


# --------------------------------------------------
# Get document by ID
# --------------------------------------------------

@router.get("/documents/{document_id}")
def get_document(
    document_id: UUID,
    store: Annotated[DocumentStore, Depends(get_document_store)],
):
    try:
        document = store.get_by_id(str(document_id))

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return document

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Failed to retrieve document %s", document_id)

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document.",
        ) from exc


# --------------------------------------------------
# Delete document
# --------------------------------------------------

@router.delete("/documents/{document_id}")
def delete_document(
    document_id: UUID,
    store: Annotated[DocumentStore, Depends(get_document_store)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
):
    document_id = str(document_id)

    try:
        # 1. Check whether document exists in PostgreSQL.
        document = store.get_by_id(document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        # 2. Delete associated vectors from Qdrant.
        vector_store.delete_by_document_id(document_id)

        # 3. Delete document metadata from PostgreSQL.
        store.delete(document_id)

        return {
            "document_id": document_id,
            "filename": document.name,
            "status": "deleted",
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document deletion failed: %s", document_id)

        raise HTTPException(
            status_code=500,
            detail="Document deletion failed. You can retry the request.",
        ) from exc
