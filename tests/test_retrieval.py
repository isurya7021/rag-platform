from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from app.vector_store.qdrant import QdrantVectorStore


embedding_model = SentenceTransformerEmbeddingModel()

vector_store = QdrantVectorStore(
    collection_name="rag_documents",
    vector_size=384,
)

# query = "What information does this document contain?"
query = "How does Retrieval-Augmented Generation help a language model answer questions?"

query_vector = embedding_model.embed(query)

results = vector_store.search(
    query_vector=query_vector,
    limit=3,
)

for result in results:
    print("\n--------------------")
    print("Score:", result.score)
    print("Content:", result.content)
    print("Metadata:", result.metadata)