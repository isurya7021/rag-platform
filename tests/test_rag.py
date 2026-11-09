
from pprint import pprint

from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from app.generation.gemini_generator import GeminiGenerator
from app.rag.service import RAGService
from app.vector_store.qdrant import QdrantVectorStore

def main():
    embedding_model = SentenceTransformerEmbeddingModel()

    vector_store = QdrantVectorStore(
        collection_name="rag_documents",
        vector_size=384,
    )

    generator = GeminiGenerator()

    rag = RAGService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        generator=generator,
    )

    result = rag.query(
        "How does RAG help a language model answer questions?",
        top_k=3,
    )

    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCES")
    pprint(result["sources"])


if __name__ == "__main__":
    main()
