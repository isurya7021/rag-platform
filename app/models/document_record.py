from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class DocumentRecord:
    id: str
    name: str
    mime_type: str
    content_hash: str
    total_chunks: int
    status: str
    created_at: datetime
    active_generation_id: UUID | None = None
    published_generation_id: UUID | None = None
