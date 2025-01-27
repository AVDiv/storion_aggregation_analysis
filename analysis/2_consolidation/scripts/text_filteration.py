import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModel:
  """A class to represent a SentenceTransformer model."""
  model: SentenceTransformer

  def __init__(self, model_name='all-MiniLM-L6-v2'):
    """Initialize the SentenceTransformer model."""
    self.model = SentenceTransformer(model_name)

  def generate_embedding(self, text):
    """Generate normalized embeddings for a list of strings."""
    embedding = self.model.encode(text)
    return embedding

  def generate_embeddings_for_corpus(self, corpus):
    """Generate normalized embeddings for a list of strings."""
    embeddings = self.model.encode(corpus)
    normalized_embeddings = np.array([emb / np.linalg.norm(emb) for emb in embeddings], dtype="float32")
    return normalized_embeddings

class FaissDB:
    """A class to manage an independent FAISS-based vector database."""

    def __init__(self, dimension):
        """Initialize a FAISS index for a given vector dimension."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance-based index
        self.embedding_documents = []  # Store metadata for embeddings

    def add_embeddings(self, docs, embeddings):
        """
        Add embeddings and associated metadata to the FAISS index.

        Args:
            docs (list): List of data corresponding to the embeddings.
            embeddings (np.array): Array of embeddings.
        """
        self.index.add(embeddings)
        for idx, emb in enumerate(embeddings):
            self.embedding_documents.append({
                "id": idx,
                "doc": docs[idx],
                "embedding": emb,
            })

    def search(self, query_embedding, k=2):
        """
        Search for the closest embeddings in the FAISS index.

        Args:
            query_embedding (np.array): Query embedding to search.
            k (int): Number of nearest neighbors to return.
        Returns:
            tuple: Distances and document of closest matches.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return distances.reshape(-1), [self.embedding_documents[index] for index in indices.reshape(-1)]

    def reset(self):
        """Delete the FAISS index and associated metadata."""
        del self.index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.embedding_documents = []
        print("FAISS database deleted.")

def calculate_pairwise_similarity(embeddings):
    """
    Calculate pairwise cosine similarity between embeddings.

    Args:
        embeddings (np.array): Array of embeddings.
    Returns:
        np.array: Pairwise similarity.
    """
    similarity_scores = cosine_similarity(embeddings, embeddings)
    return similarity_scores

def calculate_embedding_similarity(source_embedding, target_embeddings):
    """
    Calculate Cosine similarity between the source embedding and target embeddings.

    Args:
        source_embedding (np.array): Array of source embedding.
        target_embeddings (np.array): Arrays of target embeddings. 
    Returns:
        np.array: Similarity of source embedding and target embeddings.
    """
    similarity_scores = cosine_similarity(source_embedding.reshape(1, -1), target_embeddings)
    return similarity_scores

class CCF:
    """Chaining Content Filteration"""
    embed_model: EmbeddingModel
    faiss_db: FaissDB

    def __init__(self):
        """Initialize the CCF model."""
        self.embed_model = EmbeddingModel()
        self.faiss_db = FaissDB(self.embed_model.model.get_sentence_embedding_dimension())
    
    def filter_text(self, reference_text: str, victim_corpus: list[str], similarity_threshold=0.5, search_batch_size=5) -> list[str] | None:
        """
    Filter the content using the CCF model.

    Args:
        reference_text (str): Reference text.
        victim_corpus (list): List of victim corpus.
        similarity_threshold (float): Similarity threshold.
        search_batch_size (int): Search batch size.

    """
        corpus = [reference_text] + victim_corpus
        embeddings = self.embed_model.generate_embeddings_for_corpus(corpus)
        reference_text_embedding = embeddings[0, :]
        embeddings = embeddings[1:, :]
        self.faiss_db.add_embeddings(victim_corpus, embeddings)
        filtered_corpus = []
        start_doc = self.faiss_db.search(embeddings[0], k=1)[1][0]
        start_doc_similarity = calculate_embedding_similarity(reference_text_embedding, start_doc['embedding'].reshape(1, -1))
        if start_doc_similarity < similarity_threshold:
            return None
        filtered_corpus.append(start_doc['doc'])
        if start_doc['id'] > 0:
            self.faiss_db.reset()
            index_boundary = start_doc['id']
            self.faiss_db.add_embeddings(victim_corpus[:index_boundary], embeddings[:index_boundary])
            max_batches = len(victim_corpus[:index_boundary]) // search_batch_size + 1
            current_doc = start_doc.copy()
            max_depth = float('inf')
            iteration_count = 0
            max_iterations = len(victim_corpus[:index_boundary])
            while iteration_count < max_iterations:
                iteration_count = iteration_count + 1
                current_doc_embedding = current_doc['embedding']
                document_list = {}
                assumed_target_link = None
                for assumed_target_link in range(current_doc['id'] - 1, -1, -1):
                    max_depth = assumed_target_link if max_depth > assumed_target_link else max_depth
                    if not document_list.get(assumed_target_link):
                        is_result_doc_found = False
                        for batch_no in range(max_batches):
                            batch_jump_size = batch_no * search_batch_size
                            batch_results = self.faiss_db.search(current_doc_embedding, k=search_batch_size * (batch_no + 1))
                            batch_results = (batch_results[0][batch_jump_size:], batch_results[1][batch_jump_size:])
                            for dist, result_doc in zip(batch_results[0], batch_results[1]):
                                document_list[result_doc['id']] = result_doc
                                if result_doc['id'] == assumed_target_link:
                                    is_result_doc_found = True
                                    break
                            if is_result_doc_found:
                                break
                    result_doc = document_list[assumed_target_link]
                    similarity_score = calculate_embedding_similarity(current_doc_embedding, result_doc['embedding'].reshape(1, -1))
                    is_assumend_target_link_fit = similarity_score > similarity_threshold
                    if is_assumend_target_link_fit:
                        filtered_corpus.insert(0, document_list[assumed_target_link]['doc'])
                        current_doc = result_doc
                        break
                if max_depth <= 0:
                    break
        if start_doc['id'] < len(victim_corpus) - 1:
            self.faiss_db.reset()
            index_boundary = start_doc['id'] + 1
            self.faiss_db.add_embeddings(victim_corpus[index_boundary:], embeddings[index_boundary:])
            forward_corpus_size = len(victim_corpus[index_boundary:])
            max_batches = forward_corpus_size // search_batch_size + 1
            current_doc = start_doc.copy()
            current_doc['id'] = 0
            max_depth = float('-inf')
            iteration_count = 0
            max_iterations = forward_corpus_size
            while iteration_count < max_iterations:
                iteration_count = iteration_count + 1
                current_doc_embedding = current_doc['embedding']
                document_list = {}
                for assumed_target_link in range(current_doc['id'] + 1, forward_corpus_size):
                    max_depth = assumed_target_link if max_depth < assumed_target_link else max_depth
                    if not document_list.get(assumed_target_link):
                        is_result_doc_found = False
                        for batch_no in range(max_batches):
                            batch_jump_size = batch_no * search_batch_size
                            batch_results = self.faiss_db.search(current_doc_embedding, k=search_batch_size * (batch_no + 1))
                            batch_results = (batch_results[0][batch_jump_size:], batch_results[1][batch_jump_size:])
                            for dist, result_doc in zip(batch_results[0], batch_results[1]):
                                document_list[result_doc['id']] = result_doc
                                if result_doc['id'] == assumed_target_link:
                                    is_result_doc_found = True
                                    break
                            if is_result_doc_found:
                                break
                    result_doc = document_list[assumed_target_link]
                    similarity_score = calculate_embedding_similarity(current_doc_embedding, result_doc['embedding'].reshape(1, -1))
                    is_assumend_target_link_fit = similarity_score > similarity_threshold
                    if is_assumend_target_link_fit:
                        filtered_corpus.append(document_list[assumed_target_link]['doc'])
                        current_doc = result_doc
                        break
                if max_depth >= forward_corpus_size - 1:
                    break
        return filtered_corpus


splitting_pattern = r'(?<!\.)\.{2}(?!\.)|\. \.|\.{2,} \.|\n+'

def text_block_splitting(text: str) -> list[str]:
    global splitting_pattern
    return re.split(splitting_pattern, text)


class ContentFilterer():
    ccf_model: CCF
    def __init__(self) -> None:
        self.ccf_model = CCF()

    def filter_text(self, title: str, content_text: str) -> list[str]:
        text_blocks = text_block_splitting(content_text)
        filtered_text_blocks = self.ccf_model.filter_text(title, text_blocks)
        return filtered_text_blocks if filtered_text_blocks is not None else []


