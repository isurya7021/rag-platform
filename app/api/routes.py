
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.ingestion.service import IngestionService
from app.rag.service import RAGService
from google.genai import errors


router = APIRouter()

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html"}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


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

        chunks_created = service.ingest(
            str(temp_path),
            document_name=file.filename,
        )

        return {
            "filename": file.filename,
            "chunks_created": chunks_created,
            "status": "ingested",
        }

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
