from app.embeddings.base import EmbeddingModel
from app.generation.base import LLMGenerator
from app.vector_store.base import VectorStore
from app.document_store.base import DocumentStore


class RAGService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        document_store: DocumentStore,
        generator: LLMGenerator,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.document_store = document_store
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict:

        # 1. Convert the question into a vector.
        query_vector = self.embedding_model.embed(question)

        # 2. Get currently published document generations.
        published_generations = (
            self.document_store.get_published_generations()
        )

        # 3. Retrieve chunks only from published generations.
        chunks = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            published_generations=published_generations,
        )

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the documents.",
                "sources": [],
            }

        # 4. Build context from the retrieved chunks.
        context_parts = []

        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {index}]\n{chunk.content}"
            )

        context = "\n\n".join(context_parts)

        # 5. Ask Gemini to answer using the retrieved context.
        answer = self.generator.generate(
            question=question,
            context=context,
        )

        # 6. Return the answer and source metadata.
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