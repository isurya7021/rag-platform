from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingModel


class SentenceTransformerEmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()

    # def embed_batch(
    #     self,
    #     texts: list[str]
    # ) -> list[list[float]]:

    #     vectors = self.model.encode(
    #         texts,
    #         normalize_embeddings=True
    #     )

    #     return vectors.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()