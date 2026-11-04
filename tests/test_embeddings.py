from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel
)

from app.embeddings.similarity import cosine_similarity


model = SentenceTransformerEmbeddingModel()

texts = [
    "Redis is an in-memory database used for caching.",
    "Redis is commonly used as a cache in backend systems.",
    "The weather is sunny today."
]

vectors = model.embed_batch(texts)

print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))

similarity_ab = cosine_similarity(
    vectors[0],
    vectors[1]
)

similarity_ac = cosine_similarity(
    vectors[0],
    vectors[2]
)

print("Similarity A-B:", similarity_ab)
print("Similarity A-C:", similarity_ac)