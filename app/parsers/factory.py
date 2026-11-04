from pathlib import Path

from app.parsers.base import DocumentParser
from app.parsers.txt import TXTParser
from app.parsers.pdf import PDFParser
from app.parsers.docx import DOCXParser
from app.parsers.html import HTMLParser


class ParserFactory:

    _parsers: dict[str, type[DocumentParser]] = {
        ".txt": TXTParser,
        ".pdf": PDFParser,
        ".docx": DOCXParser,
        ".html": HTMLParser
    }

    @classmethod
    def get_parser(cls, file_path: str) -> DocumentParser:
        extension = Path(file_path).suffix.lower()

        parser_class = cls._parsers.get(extension)

        if parser_class is None:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return parser_class()