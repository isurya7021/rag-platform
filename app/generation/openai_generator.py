from openai import OpenAI

from app.generation.base import LLMGenerator


class OpenAIGenerator(LLMGenerator):

    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI()
        self.model = model

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Answer the user's question using only the "
                "provided context. If the context does not "
                "contain enough information, say so. "
                "Do not invent facts."
            ),
            input=(
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            ),
        )

        return response.output_text