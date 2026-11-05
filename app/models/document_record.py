from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentRecord:
    id: str
    name: str
    mime_type: str
    content_hash: str
    total_chunks: int
    status: str
    created_at: datetime