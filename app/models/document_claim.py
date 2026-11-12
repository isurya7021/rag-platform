
from dataclasses import dataclass
from uuid import UUID

from app.models.document_record import DocumentRecord


@dataclass
class DocumentClaim:
    document: DocumentRecord
    claimed: bool
    claim_token: UUID | None = None
    generation_id: UUID | None = None

