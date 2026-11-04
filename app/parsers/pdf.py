from pathlib import Path
from uuid import uuid4

import pymupdf

from app.models.document import Document, DocumentElement
from app.parsers.base import DocumentParser


class PDFParser(DocumentParser):

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)

        pdf = pymupdf.open(file_path)

        elements = []

        with pymupdf.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                elements.append(
                    DocumentElement(
                        type="text",
                        content=text,
                        metadata={
                            "page_number": page_number
                        }
                    )
                )

        return Document(
            id=str(uuid4()),
            name=path.name,
            mime_type="application/pdf",
            elements=elements
        )