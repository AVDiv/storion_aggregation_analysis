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
    return pd, pickle, tqdm


@app.cell
def _(pickle):
    with open("../df.pkl", "rb") as f:
        df = pickle.load(f)
    return df, f


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Tools testing
        """
    )
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
    mo.md(
        r"""
        ### YAKE (Looks good, but too many keywords)
        """
    )
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
    mo.md(
        r"""
        ### Keybert (Useless)
        """
    )
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
    entities = fair_ner_pipe(text)
    return (entities,)


@app.cell
def _(entities):
    entities
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Approach
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
