from app.utils.file_hash import calculate_file_hash
from app.vector_store.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(
    collection_name="rag_documents",
    vector_size=384,
)

file_hash = calculate_file_hash("documents/sample.pdf")

document_id = vector_store.find_document_id_by_hash(file_hash)

print("Existing document ID:", document_id)