import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import string
import csv
nltk.download("stopwords")

GLOVE_PATH = "data/glove.6B.300d.txt"
NO_GLOVE_DIMENSIONS = 300
PREPROCESSING_METHODS = ["glove"]


def tokenize(text: str, preprocessing: str) -> list:
    # lowercasing and tokenization
    text = word_tokenize(text)
    text = map(lambda sample: sample.lower(), text)

    # stopword removal
    stop_words = set(stopwords.words("english"))
    text = list(filter(lambda sample: sample not in stop_words, text))

    # punctuation removal
    text = list(filter(lambda word: word not in string.punctuation, text))

    # stemming
    if not preprocessing == "glove":
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]

    return text


def get_glove_embeddings() -> dict:
    df = pd.read_csv("data/glove.6B.300d.txt", header=None, sep=" ", quoting=csv.QUOTE_NONE)
    glove = {}

    for i in df.index:
        glove[df[0][i]] = list(df.iloc[i, 1:])

    return glove


def vectorize(text: list, embeddings: dict) -> list:
    vectorized = [0] * NO_GLOVE_DIMENSIONS

    for word in text:
        if word in embeddings:
            word_embedding = embeddings[word]

            for i in range(NO_GLOVE_DIMENSIONS):
                vectorized[i] += word_embedding[i]

    return vectorized


def read_data(file: str, preprocessing: str) -> (list, list):
    assert preprocessing in PREPROCESSING_METHODS, f"Preprocessing method \"{preprocessing}\" is not supported!"

    df = pd.read_csv(file, quoting=csv.QUOTE_NONE)
    X = []
    y = []
    embeddings = {}

    if preprocessing == "glove":
        embeddings = get_glove_embeddings()

    for i in df.index:
        sample = tokenize(df["text"][i], preprocessing)

        if preprocessing == "glove":
            sample = vectorize(sample, embeddings)

        X.append(sample)
        y.append(df["label"][i])

    return X, y


def write_preds(file: str, preds: list):
    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])

        writer.writeheader()

        for i, pred in enumerate(preds):
            writer.writerow({"id": i, "label": pred})
