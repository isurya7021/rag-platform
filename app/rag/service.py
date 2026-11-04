
from app.embeddings.base import EmbeddingModel
from app.generation.base import LLMGenerator
from app.vector_store.base import VectorStore


class RAGService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        generator: LLMGenerator,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict:

        # 1. Convert the question into a vector.
        query_vector = self.embedding_model.embed(question)

        # 2. Retrieve the most relevant chunks.
        chunks = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
        )

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the documents.",
                "sources": [],
            }

        # 3. Build context from the retrieved chunks.
        context_parts = []

        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {index}]\n{chunk.content}"
            )

        context = "\n\n".join(context_parts)

        # 4. Ask Gemini to answer using the retrieved context.
        answer = self.generator.generate(
            question=question,
            context=context,
        )

        # 5. Return the answer and source metadata.
        sources = [
            {
                "chunk_id": chunk.id,
                "document_name": chunk.metadata.get("document_name"),
                "document_id": chunk.metadata.get("document_id"),
                "score": chunk.score,
                "content": chunk.content,
            }
            for chunk in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
