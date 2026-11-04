from pathlib import Path
from uuid import uuid4

from docx import Document as DocxDocument

from app.models.document import Document, DocumentElement
from app.parsers.base import DocumentParser


class DOCXParser(DocumentParser):

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)

        docx = DocxDocument(file_path)

        elements = []

        for paragraph in docx.paragraphs:
            text = paragraph.text.strip()

            if not text:
                continue

            element_type = "heading" if paragraph.style.name.startswith("Heading") else "text"

            elements.append(
                DocumentElement(
                    type=element_type,
                    content=text,
                    metadata={
                        "style": paragraph.style.name
                    }
                )
            )

        return Document(
            id=str(uuid4()),
            name=path.name,
            mime_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            elements=elements
        )