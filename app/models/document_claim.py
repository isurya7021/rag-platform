
from dataclasses import dataclass

from app.models.document_record import DocumentRecord


@dataclass
class DocumentClaim:
    document: DocumentRecord
    claimed: bool
