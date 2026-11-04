from abc import ABC, abstractmethod


class LLMGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        pass