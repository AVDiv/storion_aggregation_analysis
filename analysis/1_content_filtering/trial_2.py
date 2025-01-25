import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Trial 2

    This is an enhanced version of trial 1
    """)
    return


@app.cell
def _():
    # Import the necessary libraries
    import numpy as np
    import pandas as pd
    import pickle
    import re
    return np, pd, pickle, re


@app.cell
def _(pickle):
    # Load the data
    with open('../df.pkl', 'rb') as f:
        df = pickle.load(f)
    return df, f


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Approach")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Select some text samples for testing")
    return


@app.cell
def _(df, np):
    np.random.seed(0)
    sample_ids = np.random.choice(df.index, 5)
    sample_records = df.loc[sample_ids]
    return sample_ids, sample_records


@app.cell
def _(sample_records):
    # URLs
    sample_records['url'].to_list()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Let's split the text into contextually meaningful blocks, which talks about a single thing, Probably something like a paragraph")
    return


@app.cell
def _(sample_records):
    _text = sample_records.iloc[0]['content']
    _text
    return


@app.cell
def _(sample_records):
    _text = sample_records.iloc[1]['content']
    _text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Seems like there are weird period patterns in between such text block, let's use that, and \\n as a delimiter to split the text into blocks.")
    return


@app.cell
def _():
    pattern = r'(?<!\.)\.{2}(?!\.)|\. \.|\.{2,} \.|\n+'
    return (pattern,)


@app.cell
def _(pattern, re, sample_records):
    _text = sample_records.iloc[0]['content']
    re.split(pattern, _text)
    return


@app.cell
def _(pattern, re, sample_records):
    _text = sample_records.iloc[1]['content']
    re.split(pattern, _text)
    return


@app.cell
def _(pattern, re, sample_records):
    _text = sample_records.iloc[2]['content']
    re.split(pattern, _text)
    return


@app.cell
def _(pattern, re, sample_records):
    _text = sample_records.iloc[3]['content']
    re.split(pattern, _text)
    return


@app.cell
def _(pattern, re, sample_records):
    _text = sample_records.iloc[4]['content']
    re.split(pattern, _text)
    return


@app.cell
def _(pattern, re, sample_records):
    content_corpuses = [re.split(pattern, text) for text in sample_records['content']]
    return (content_corpuses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("Now, we can try the chaining process")
    return


@app.cell
def _():
    # Import the necessary libraries
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity
    return SentenceTransformer, cosine_similarity, faiss


@app.cell
def _(SentenceTransformer, np):
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
            normalized_embeddings = np.array([emb / np.linalg.norm(emb) for emb in embeddings], dtype='float32')
            return normalized_embeddings
    return (EmbeddingModel,)


@app.cell
def _(faiss):
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
    return (FaissDB,)


@app.cell
def _(cosine_similarity):
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
    return calculate_embedding_similarity, calculate_pairwise_similarity


@app.cell
def _(EmbeddingModel, FaissDB, calculate_embedding_similarity, np):
    class C3F:
        '''Collective Chaining Content Filteration'''
        embed_model: EmbeddingModel
        faiss_db: FaissDB

        def __init__(self):
            """Initialize the C3F model."""
            self.embed_model = EmbeddingModel()
            self.faiss_db = FaissDB(self.embed_model.model.get_sentence_embedding_dimension())

        def _calculate_weighted_similarity(self, title_embedding, chained_embeddings, new_text_embedding, similarity_threshold):
            """
            Calculate weighted similarity between new text and chained content.
            
            Args:
                title_embedding (np.array): Embedding of the title.
                chained_embeddings (np.array): Embeddings of chained content.
                new_text_embedding (np.array): Embedding of new text.
                similarity_threshold (float): Similarity threshold.
            
            Returns:
                float: Weighted similarity score.
            """
            # Calculate title similarity with 30% weight
            title_similarity = calculate_embedding_similarity(new_text_embedding, title_embedding.reshape(1, -1))[0][0]
            title_weighted_similarity = title_similarity * 0.3

            # Calculate weighted similarity for chained content
            if len(chained_embeddings) == 0:
                return title_weighted_similarity

            # Calculate cosine similarities
            content_similarities = calculate_embedding_similarity(new_text_embedding, chained_embeddings).flatten()

            # Create linear weights for chained content (closer texts get higher weights)
            linear_weights = np.linspace(1, 0.1, len(chained_embeddings))
            linear_weights /= linear_weights.sum()  # Normalize weights

            # Calculate weighted content similarity
            content_weighted_similarity = np.dot(content_similarities, linear_weights) * 0.7

            # Combine title and content weighted similarities
            total_weighted_similarity = title_weighted_similarity + content_weighted_similarity

            return total_weighted_similarity

        def filtertext_approach_1(self, referencetext: str, _victim_corpus: list[str], similarity_threshold=0.5):
            """
            Filter the content using the C3F model with enhanced chaining logic.

            Args:
                referencetext (str): Reference text.
                _victim_corpus (list): List of victim corpus.
                similarity_threshold (float): Similarity threshold.

            Returns:
                list: Filtered corpus.
            """
            # Generate embeddings
            corpus = [referencetext] + _victim_corpus
            embeddings = self.embed_model.generate_embeddings_for_corpus(corpus)
            
            # Extract embeddings
            referencetext_embedding = embeddings[0, :]
            embeddings = embeddings[1:, :]

            # Initialize variables
            filtered_corpus = []
            reference_to_victim_similarity = calculate_embedding_similarity(referencetext_embedding, embeddings).flatten()
            start_doc_idx = np.argmax(reference_to_victim_similarity)

            # Check initial similarity
            if reference_to_victim_similarity[start_doc_idx] < similarity_threshold:
                return None

            # Add starting document
            start_doc = _victim_corpus[start_doc_idx]
            filtered_corpus.append(start_doc)

            # Determine possible chaining directions
            backward_possible = start_doc_idx > 0
            forward_possible = start_doc_idx < len(_victim_corpus) - 1

            # Alternate chaining directions
            current_doc = start_doc_idx
            current_embedding = embeddings[current_doc, :]
            chain_status = {
                'backward': backward_possible,
                'forward': forward_possible
            }
            direction_toggle = True  # True for backward, False for forward

            while chain_status['backward'] or chain_status['forward']:
                if direction_toggle and chain_status['backward']:
                    # Backward chaining
                    search_range = range(current_doc - 1, -1, -1)
                    insert_method = filtered_corpus.insert
                    insert_index = 0
                elif chain_status['forward']:
                    # Forward chaining
                    search_range = range(current_doc + 1, len(_victim_corpus))
                    insert_method = filtered_corpus.append
                    insert_index = 0  # not used for append

                # Attempt chaining in the current direction
                chained = False
                for assumed_target_link in search_range:
                    # Calculate weighted similarity
                    weighted_similarity = self._calculate_weighted_similarity(
                        referencetext_embedding, 
                        np.array([emb for emb in [embeddings[idx, :] for idx in range(len(filtered_corpus))]]), 
                        embeddings[assumed_target_link, :], 
                        similarity_threshold
                    )

                    # Check if similarity meets threshold
                    if weighted_similarity > similarity_threshold:
                        # Insert or append the document
                        if direction_toggle:
                            filtered_corpus.insert(0, _victim_corpus[assumed_target_link])
                        else:
                            filtered_corpus.append(_victim_corpus[assumed_target_link])
                        
                        current_doc = assumed_target_link
                        current_embedding = embeddings[current_doc, :]
                        chained = True
                        break

                # Update chain status and direction
                if direction_toggle:
                    chain_status['backward'] = not chained and current_doc > 0
                else:
                    chain_status['forward'] = not chained and current_doc < len(_victim_corpus) - 1

                direction_toggle = not direction_toggle

            return filtered_corpus

        def filtertext_approach_2(self, referencetext: str, _victim_corpus: list[str], similarity_threshold=0.5, search_batch_size=5):
            """
            Filter the content using the C3F model with enhanced chaining logic.

            Args:
                referencetext (str): Reference text.
                _victim_corpus (list): List of victim corpus.
                similarity_threshold (float): Similarity threshold.
                search_batch_size (int): Search batch size.

            Returns:
                list: Filtered corpus.
            """
            # Generate embeddings
            corpus = [referencetext] + _victim_corpus
            embeddings = self.embed_model.generate_embeddings_for_corpus(corpus)
            
            # Extract embeddings
            referencetext_embedding = embeddings[0, :]
            embeddings = embeddings[1:, :]

            # Reset and add embeddings to FAISS DB
            self.faiss_db.reset()
            self.faiss_db.add_embeddings(_victim_corpus, embeddings)

            # Find starting point
            start_doc = self.faiss_db.search(referencetext_embedding, k=1)[1][0]
            start_doc_similarity = calculate_embedding_similarity(referencetext_embedding, start_doc['embedding'].reshape(1, -1))

            # Check initial similarity
            if start_doc_similarity < similarity_threshold:
                return None

            # Initialize filtered corpus
            filtered_corpus = [start_doc['doc']]

            # Determine possible chaining directions
            backward_possible = start_doc['id'] > 0
            forward_possible = start_doc['id'] < len(_victim_corpus) - 1

            # Alternate chaining directions
            current_doc = start_doc
            chain_status = {
                'backward': backward_possible,
                'forward': forward_possible
            }
            direction_toggle = True  # True for backward, False for forward

            while chain_status['backward'] or chain_status['forward']:
                if direction_toggle and chain_status['backward']:
                    # Backward chaining
                    corpus_slice = _victim_corpus[:current_doc['id']]
                    embeddings_slice = embeddings[:current_doc['id']]
                    search_range = range(current_doc['id'] - 1, -1, -1)
                    insert_method = filtered_corpus.insert
                    insert_index = 0
                elif chain_status['forward']:
                    # Forward chaining
                    corpus_slice = _victim_corpus[current_doc['id'] + 1:]
                    embeddings_slice = embeddings[current_doc['id'] + 1:]
                    search_range = range(0, len(corpus_slice))
                    insert_method = filtered_corpus.append
                    insert_index = 0  # not used for append

                # Reset FAISS DB for the current slice
                self.faiss_db.reset()
                self.faiss_db.add_embeddings(corpus_slice, embeddings_slice)

                # Attempt chaining in the current direction
                chained = False
                max_batches = len(corpus_slice) // search_batch_size + 1
                document_list = {}

                for assumed_target_link in search_range:
                    # Calculate weighted similarity
                    weighted_similarity = self._calculate_weighted_similarity(
                        referencetext_embedding, 
                        np.array([emb for emb in [embeddings[idx, :] for idx in range(len(filtered_corpus))]]), 
                        embeddings_slice[assumed_target_link, :], 
                        similarity_threshold
                    )

                    # Check if similarity meets threshold
                    if weighted_similarity > similarity_threshold:
                        # Perform batch search to find the document
                        is_result_doc_found = False
                        for batch_no in range(max_batches):
                            batch_jump_size = batch_no * search_batch_size
                            batch_results = self.faiss_db.search(current_doc['embedding'], k=search_batch_size * (batch_no + 1))
                            batch_results = (batch_results[0][batch_jump_size:], batch_results[1][batch_jump_size:])
                            
                            for dist, result_doc in zip(batch_results[0], batch_results[1]):
                                document_list[result_doc['id']] = result_doc
                                if result_doc['id'] == assumed_target_link:
                                    is_result_doc_found = True
                                    break
                            
                            if is_result_doc_found:
                                break

                        # Insert or append the document
                        if direction_toggle:
                            filtered_corpus.insert(0, corpus_slice[assumed_target_link])
                        else:
                            filtered_corpus.append(corpus_slice[assumed_target_link])
                        
                        current_doc = result_doc
                        chained = True
                        break

                # Update chain status and direction
                if direction_toggle:
                    chain_status['backward'] = not chained and current_doc['id'] > 0
                else:
                    chain_status['forward'] = not chained and current_doc['id'] < len(_victim_corpus) - 1

                direction_toggle = not direction_toggle

            return filtered_corpus
    return (C3F,)


@app.cell
def _():
    import difflib

    def highlight_corpus_difference(original_corpus: list[str], new_corpus: list[str]):
      # Print the new corpus with highlights(red text for removed)
      diff = difflib.unified_diff(original_corpus, new_corpus, lineterm="")
      for line in diff:
          print(f"{'\033[31m' if line.startswith('-') else '\033[0m'}{line}")
    return difflib, highlight_corpus_difference


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Evaluation

    Let's evaluate the method
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Sample 1")
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _c3f = C3F()
    _reference_text = sample_records.iloc[0]['title']
    _victim_corpus = content_corpuses[0]
    _filtered_corpus = _c3f.filtertext_approach_1(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _c3f = C3F()
    _reference_text = sample_records.iloc[0]['title']
    _victim_corpus = content_corpuses[0]
    _filtered_corpus = _c3f.filtertext_approach_2(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Sample 2")
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[1]['title']
    _victim_corpus = content_corpuses[1]
    _filtered_corpus = _ccf.filtertext_approach_1(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[1]['title']
    _victim_corpus = content_corpuses[1]
    _filtered_corpus = _ccf.filtertext_approach_2(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Sample 3")
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[2]['title']
    _victim_corpus = content_corpuses[2]
    _filtered_corpus = _ccf.filtertext_approach_1(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[2]['title']
    _victim_corpus = content_corpuses[2]
    _filtered_corpus = _ccf.filtertext_approach_2(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Sample 4")
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[3]['title']
    _victim_corpus = content_corpuses[3]
    _filtered_corpus = _ccf.filtertext_approach_1(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[3]['title']
    _victim_corpus = content_corpuses[3]
    _filtered_corpus = _ccf.filtertext_approach_2(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Sample 5")
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[4]['title']
    _victim_corpus = content_corpuses[4]
    _filtered_corpus = _ccf.filtertext_approach_1(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell
def _(C3F, content_corpuses, highlight_corpus_difference, sample_records):
    _ccf = C3F()
    _reference_text = sample_records.iloc[4]['title']
    _victim_corpus = content_corpuses[4]
    _filtered_corpus = _ccf.filtertext_approach_2(_reference_text, _victim_corpus, similarity_threshold=0)
    print()
    print(_reference_text)
    print()
    highlight_corpus_difference(_victim_corpus, _filtered_corpus)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Compute Performance Metrics")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Let's benchmark the performance of the methods")
    return


@app.cell
def _(C3F, content_corpuses, np, pd, sample_ids, sample_records):
    import time
    import psutil

    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss

    def get_time_taken(func, *args):
        start_time = time.time()
        func(*args)
        return time.time() - start_time
    approach_1_time = []
    approach_1_memory = []
    approach_2_time = []
    approach_2_memory = []
    for i in range(5):
        _reference_text = sample_records.iloc[i]['title']
        _victim_corpus = content_corpuses[i]
        _ccf = C3F()
        approach_1_time.append(get_time_taken(_ccf.filtertext_approach_1, _reference_text, _victim_corpus, 0))
        approach_1_memory.append(get_memory_usage())
        _ccf = C3F()
        approach_2_time.append(get_time_taken(_ccf.filtertext_approach_2, _reference_text, _victim_corpus, 0))
        approach_2_memory.append(get_memory_usage())
    approach_1_time.append(np.mean(approach_1_time))
    approach_1_memory.append(np.mean(approach_1_memory))
    approach_2_time.append(np.mean(approach_2_time))
    approach_2_memory.append(np.mean(approach_2_memory))
    bench_df = pd.DataFrame({'Sample ID': sample_ids.tolist() + ['Average'], 'Approach 1 Time (s)': approach_1_time, 'Approach 1 Memory (MB)': [mem / 1000000.0 for mem in approach_1_memory], 'Approach 2 Time (s)': approach_2_time, 'Approach 2 Memory (MB)': [mem / 1000000.0 for mem in approach_2_memory]})
    bench_df
    return (
        approach_1_memory,
        approach_1_time,
        approach_2_memory,
        approach_2_time,
        bench_df,
        get_memory_usage,
        get_time_taken,
        i,
        psutil,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("By manual intervention, approach 2 seems to be the best in accuracy, but on time-wise & memory-wise, approach 1 shows better results. So, method 2 is the best for this problem, as there is no much loss of compute performance relative to approach 1.")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
