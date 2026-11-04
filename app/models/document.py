from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentElement:
    type: str
    content: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    id: str
    name: str
    mime_type: str
    elements: list[DocumentElement] = field(default_factory=list)