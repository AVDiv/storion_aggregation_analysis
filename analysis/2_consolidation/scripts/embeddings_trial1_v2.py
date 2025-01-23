import numpy as np
from sentence_transformers import SentenceTransformer


class Embeddings_v2:
    model: SentenceTransformer

    def __init__(self):
        model_name = "all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, *, article_content):
        embedding = None
        embedding = self.model.encode(article_content)
        return embedding
