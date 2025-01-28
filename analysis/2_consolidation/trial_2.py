import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Trial 2 
        something different, maybe. Hope this works! :')
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    from tqdm import tqdm
    import pickle
    import numpy as np
    return np, pd, pickle, tqdm


@app.cell
def _(pickle):
    with open("../df.pkl", "rb") as f:
        df = pickle.load(f)
    return df, f


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tools testing""")
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from yake import KeywordExtractor
    from keybert import KeyBERT
    from transformers import pipeline
    return KeyBERT, KeywordExtractor, TfidfVectorizer, pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### YAKE (Looks good, but too many keywords)""")
    return


@app.cell
def _(KeywordExtractor):
    kw_extractor = KeywordExtractor()
    return (kw_extractor,)


@app.cell
def _(df):
    text = df.loc[df["event_id"] == "case_1"].iloc[0]["content"]
    return (text,)


@app.cell
def _(kw_extractor, text):
    keywords = kw_extractor.extract_keywords(text)
    return (keywords,)


@app.cell
def _(keywords):
    keywords
    return


@app.cell
def _(keywords):
    kw_sum = 0
    kw_sum = sum([keyword[1] for keyword in keywords])
    kw_sum
    return (kw_sum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Keybert (Useless)""")
    return


@app.cell
def _(KeyBERT):
    model = KeyBERT(model="all-mpnet-base-v2")
    return (model,)


@app.cell
def _(model, text):
    keywords_kb = model.extract_keywords(text)
    keywords_kb
    return (keywords_kb,)


@app.cell
def _(mo):
    mo.md(r"""### NER_xlm-roberta-large-finetuned-conll03-english""")
    return


@app.cell
def _(pipeline):
    fair_ner_pipe = pipeline("token-classification", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english")
    return (fair_ner_pipe,)


@app.cell
def _(fair_ner_pipe, text):
    raw_entities = fair_ner_pipe(text)
    return (raw_entities,)


@app.cell
def _(text):
    text
    return


@app.cell
def _(raw_entities):
    raw_entities
    return


@app.cell
def _():
    def combine_same_entities(_text, _raw_entities):
        prev_segment = None
        entities = []
        for segment in _raw_entities:
            segment['word'] = segment['word'].replace('▁', ' ')
            original_entity_word = segment['word']
            entity_word = segment['word'].rstrip()
            segment['end'] = segment['end'] - (len(original_entity_word) - len(entity_word))
            if segment['word'].isspace():
                continue
            appended_to_prev_segment = False
            if prev_segment is not None and prev_segment['entity'] == segment['entity']:
                if prev_segment['end'] == segment['start']:
                    entities[-1]['word'] += segment['word']
                    appended_to_prev_segment = True
                elif _text[prev_segment['end']:segment['start']].isspace():
                    entities[-1]['word'] += _text[prev_segment['end']:segment['start']] + segment['word']
                    appended_to_prev_segment = True

                if appended_to_prev_segment:
                    entities[-1]['end'] = segment['end']
                    entities[-1]['score'] = (entities[-1]['score'] + segment['score'])/2

            if not appended_to_prev_segment:
                original_entity_word = entity_word
                entity_word = entity_word.lstrip()
                segment['start'] = segment['start'] + (len(original_entity_word) - len(entity_word))
                entities.append({
                    'entity': segment['entity'],
                    'word': entity_word,
                    'score': segment['score'],
                    'start': segment['start'],
                    'end': segment['end']
                })
            prev_segment = segment.copy()
        return entities
    return (combine_same_entities,)


@app.cell
def _(combine_same_entities, raw_entities, text):
    combine_same_entities(text, raw_entities)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Approach""")
    return


@app.cell
def _():
    from scripts.text_filteration import ContentFilterer
    return (ContentFilterer,)


@app.cell
def _(ContentFilterer):
    content_filter = ContentFilterer()
    return (content_filter,)


@app.cell
def _(df):
    selected_df = df.iloc[:200]
    selected_df = selected_df._append(df.loc[df['event_id'] == 'case_1'])
    selected_df = selected_df._append(df.loc[df['event_id'] == 'case_2'])
    selected_df = selected_df._append(df.loc[df['event_id'] == 'case_3'])
    return (selected_df,)


@app.cell
def _(selected_df):
    selected_df.tail()
    return


@app.cell
def _(content_filter, selected_df, tqdm):
    cleaned_contents = []

    # For cleaning all the contents
    # for idx, row in tqdm(df.iterrows(), desc="Filtering contents with CCF...", total=len(df)):
    #     cleaned_content = content_filter.filter_text(row['title'], row['content'])
    #     cleaned_contents.append(cleaned_content)

    # For cleaning of selected sample set
    for idx, row in tqdm(selected_df.iterrows(), desc="Filtering contents with CCF...", total=len(selected_df)):
        cleaned_content = content_filter.filter_text(row['title'], row['content'])
        cleaned_contents.append(cleaned_content)
    return cleaned_content, cleaned_contents, idx, row


@app.cell
def _(KeywordExtractor, pipeline):
    # Keyword extractor & NER Model
    yake_kw = KeywordExtractor()
    ner_pipe = pipeline("token-classification", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english")
    return ner_pipe, yake_kw


@app.cell
def _(cleaned_contents, combine_same_entities, ner_pipe, yake_kw):
    # Sample of keyword extraction & NER
    _sample_text = " ".join(cleaned_contents[19])
    print(_sample_text)
    sample_keywords = yake_kw.extract_keywords(_sample_text)
    print(sample_keywords)
    sample_entities = combine_same_entities(_sample_text, ner_pipe(_sample_text))
    sample_entities
    return sample_entities, sample_keywords


@app.cell
def _():
    def filter_non_entity_keywords(_keywords, _entities):
        entity_list = set([entity['word'] for entity in _entities])
        _keywords = [keyword for keyword in _keywords if keyword[0] not in entity_list]
        return _keywords
    return (filter_non_entity_keywords,)


@app.cell
def _(np):
    from difflib import SequenceMatcher
    from collections import defaultdict

    # Function to calculate similarity between two strings
    def calculate_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Function to create a similarity matrix for all words
    def create_similarity_matrix(word_list):
        size = len(word_list)
        matrix = [[0] * size for _ in range(size)]
        
        for i, word1 in enumerate(word_list):
            for j, word2 in enumerate(word_list):
                if i != j:  # Avoid self-comparison
                    matrix[i][j] = calculate_similarity(word1, word2)
        return matrix

    # Function to group words based on similarity matrix
    def group_similar_words(word_list, top_values_threshold=None):
        similarity_matrix = np.array(create_similarity_matrix(word_list))
        # Dynamic thresholding 
        top_n_values = len(similarity_matrix[similarity_matrix>top_values_threshold]) if top_values_threshold is not None else similarity_matrix.shape[0]//1
        top_n_values = 1 if top_n_values==0 else top_n_values
        top_n_values *= -1
        top_n_similarities = np.partition(similarity_matrix.flatten(), top_n_values)[top_n_values:]
        threshold = np.mean(top_n_similarities)
        print(f"Threshold set to {threshold}")
        groups = []
        visited = set()

        for i, word in enumerate(word_list):
            if word not in visited:
                group = set()
                for j, similarity in enumerate(similarity_matrix[i]):
                    if similarity >= threshold:
                        group.add(word_list[j])
                group.add(word)  # Include the current word
                groups.append(group)
                visited.update(group)  # Mark all words in this group as visited

        return groups
    return (
        SequenceMatcher,
        calculate_similarity,
        create_similarity_matrix,
        defaultdict,
        group_similar_words,
    )


@app.cell
def _(group_similar_words, sample_keywords):
    group_similar_words([keyword[0] for keyword in sample_keywords], top_values_threshold=0.3)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
