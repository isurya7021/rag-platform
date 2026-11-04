
from app.generation.gemini_generator import GeminiGenerator


generator = GeminiGenerator()

answer = generator.generate(
    question="What is RAG?",
    context=(
        "Retrieval-Augmented Generation (RAG) retrieves "
        "relevant information from external sources and "
        "provides it to a language model as context."
    ),
)

print("Gemini response:")
print(answer)
