
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.ingestion.service import IngestionService
from app.rag.service import RAGService
from google.genai import errors
from app.document_store.base import DocumentStore
from app.vector_store.base import VectorStore


router = APIRouter()

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html"}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service

def get_document_store(request: Request) -> DocumentStore:
    return request.app.state.document_store

def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


@router.post("/documents")
async def ingest_document(
    file: Annotated[UploadFile, File()],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
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

            while contents := await file.read(1024 * 1024):
                temp_file.write(contents)

        result = service.ingest(
            str(temp_path),
            document_name=file.filename,
        )

        return result

    except Exception as exc:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


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
        raise HTTPException(
            status_code=exc.code,
            detail=f"LLM provider error: {exc.message}",
        ) from exc

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {type(exc).__name__}: {exc}",
        ) from exc
    


@router.get("/documents")
def list_documents(
    store: Annotated[DocumentStore, Depends(get_document_store)],
):
    try:
        documents = store.list_all()
        return documents

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents: {type(exc).__name__}: {exc}",
        ) from exc


@router.get("/documents/{document_id}")
def get_document(
    document_id: str,
    store: Annotated[DocumentStore, Depends(get_document_store)],
):
    document = store.get_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return document


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    store: Annotated[DocumentStore, Depends(get_document_store)],
    vector_store: Annotated[
        VectorStore,
        Depends(get_vector_store),
    ],
):
    # 1. Check whether the document exists.
    document = store.get_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    try:
        # 2. Delete all associated chunks.
        vector_store.delete_by_document_id(document_id)

        # 3. Delete document metadata.
        store.delete(document_id)

        return {
            "document_id": document_id,
            "filename": document.name,
            "status": "deleted",
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {type(exc).__name__}: {exc}",
        ) from exc