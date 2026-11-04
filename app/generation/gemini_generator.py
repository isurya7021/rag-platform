import os

from dotenv import load_dotenv
from google import genai

from app.generation.base import LLMGenerator
from google.genai import errors


load_dotenv()


class GeminiGenerator(LLMGenerator):

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Set it in your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
    Answer the question using only the provided context.
    If the context does not contain enough information,
    say that you cannot determine the answer from it.
    Do not invent facts.

    Context:
    {context}

    Question:
    {question}
    """

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text

        except errors.APIError as exc:
            print(
                f"Gemini API error: HTTP {exc.code}: {exc.message}",
                flush=True,
            )
            raise