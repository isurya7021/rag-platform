from pathlib import Path
from uuid import uuid4

from app.models.document import Document, DocumentElement
from app.parsers.base import DocumentParser


class TXTParser(DocumentParser):

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)

        text = path.read_text(encoding="utf-8")

        element = DocumentElement(
            type="text",
            content=text,
            metadata={}
        )

        return Document(
            id=str(uuid4()),
            name=path.name,
            mime_type="text/plain",
            elements=[element]
        )