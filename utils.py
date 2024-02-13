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


def tokenize(text: str) -> list:
    # lowercasing and tokenization
    text = word_tokenize(text)
    text = map(lambda sample: sample.lower(), text)

    # stemming
    # stemmer = PorterStemmer()
    # text = [stemmer.stem(word) for word in text]

    # stopword removal
    stop_words = set(stopwords.words("english"))
    text = list(filter(lambda sample: sample not in stop_words, text))

    # punctuation removal
    text = list(filter(lambda word: word not in string.punctuation, text))

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


def read_data(file: str) -> (list, list):
    df = pd.read_csv(file)
    X = []
    y = []
    max_length = 0
    embeddings = get_glove_embeddings()

    for i in df.index:
        tokenized = tokenize(df["text"][i])
        vectorized = vectorize(tokenized, embeddings)

        X.append(vectorized)
        y.append(df["label"][i])
        max_length = max(max_length, len(vectorized))

    return X, y
