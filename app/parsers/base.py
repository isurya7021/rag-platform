from abc import ABC, abstractmethod

from app.models.document import Document


class DocumentParser(ABC):

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        pass