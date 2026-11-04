from pathlib import Path
from uuid import uuid4

from bs4 import BeautifulSoup

from app.models.document import Document, DocumentElement
from app.parsers.base import DocumentParser


class HTMLParser(DocumentParser):

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)

        html = path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")

        elements = []

        for element in soup.find_all(["h1", "h2", "h3", "p"]):

            text = element.get_text(" ", strip=True)

            if not text:
                continue

            if element.name in ["h1", "h2", "h3"]:
                element_type = "heading"
            else:
                element_type = "text"

            elements.append(
                DocumentElement(
                    type=element_type,
                    content=text,
                    metadata={
                        "tag": element.name
                    }
                )
            )

        return Document(
            id=str(uuid4()),
            name=path.name,
            mime_type="text/html",
            elements=elements
        )