from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import csv
import math

nltk.download("stopwords")

GLOVE_PATH = "data/glove.6B.300d.txt"
NO_GLOVE_DIMENSIONS = 300
PREPROCESSING_METHODS = ["glove", "glove_rtf_igm", "pretrained"]
IGM_LAMBDA = 7.0


def tokenize(text: str) -> list:
    # lowercasing and tokenization
    text = word_tokenize(text)
    text = map(lambda sample: sample.lower(), text)

    # stopword removal
    stop_words = set(stopwords.words("english"))
    text = list(filter(lambda sample: sample not in stop_words, text))

    # removal of words with non-alpha characters
    text = list(filter(lambda word: word.isalpha(), text))

    return text


def igm(word: str, counts: dict, no_words_per_label: list) -> float:
    if (word not in counts[0] or counts[0][word] == 1) and (word not in counts[1] or counts[1][word] == 1):
        a_b = 1  # avoiding a_b values < 1 when word occurs only once or 0 times in both classes

    elif word not in counts[0]:
        a_b = (counts[1][word] / no_words_per_label[1]) / (1 / no_words_per_label[0])

    elif word not in counts[1]:
        a_b = (counts[0][word] / no_words_per_label[0]) / (1 / no_words_per_label[1])

    else:
        a_b = max((counts[0][word] / no_words_per_label[0]) / (counts[1][word] / no_words_per_label[1]),
                  (counts[1][word] / no_words_per_label[1]) / (counts[0][word] / no_words_per_label[0]))

    return 1 + IGM_LAMBDA * a_b


def get_rtf_igm_weights(file: str, cache=None, prob_per_word=None) -> (list, dict):
    print("Calculating rtf-igm weights")

    start_time = datetime.now()
    df = pd.read_csv(file, quoting=csv.QUOTE_NONE)
    stemmer = PorterStemmer()
    stems = []
    labels = []
    stem_counts = {0: {}, 1: {}}

    for i in df.index:
        sample = tokenize(df["text"][i])
        sample_stems = []

        for word in sample:
            stemmed_word = stemmer.stem(word)
            sample_stems.append(stemmed_word)

            if stemmed_word in stem_counts[df["label"][i]]:
                stem_counts[df["label"][i]][stemmed_word] += 1

            else:
                stem_counts[df["label"][i]][stemmed_word] = 1

        stems.append(sample_stems)
        labels.append(df["label"][i])

    weights = []
    no_words_per_label = [sum(stem_counts[0].values()), sum(stem_counts[1].values())]

    if prob_per_word is None:
        prob_per_word = {}

    for sample in stems:
        sample_weights = []

        for word in sample:
            if word not in prob_per_word:
                prob_per_word[word] = igm(word, stem_counts, no_words_per_label)

            sample_weights.append(math.sqrt(sample.count(word) / len(sample)) * prob_per_word[word])

        weights.append(sample_weights)

    if cache is not None:
        with open(cache, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["word", "weight"])

            writer.writeheader()

            for key in prob_per_word.keys():
                writer.writerow({"word": key, "weight": prob_per_word[key]})

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return weights, prob_per_word


def get_rtf_igm_test_weights(file: str, prob_per_word: dict) -> list:
    print("Calculating rtf-igm test weights")

    start_time = datetime.now()
    df = pd.read_csv(file, quoting=csv.QUOTE_NONE)
    stemmer = PorterStemmer()
    weights = []
    # when we have no "igm" information, we consider the word to be occurring with equal frequency in both classes
    default_prob = 1 + IGM_LAMBDA

    for i in df.index:
        sample = tokenize(df["text"][i])
        sample = [stemmer.stem(word) for word in sample]
        sample_weights = []

        for word in sample:
            if word in prob_per_word:
                sample_weights.append(math.sqrt(sample.count(word) / len(sample)) * prob_per_word[word])

            else:
                sample_weights.append(math.sqrt(sample.count(word) / len(sample)) * default_prob)

        weights.append(sample_weights)

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return weights


def get_glove_embeddings() -> dict:
    df = pd.read_csv("data/glove.6B.300d.txt", header=None, sep=" ", quoting=csv.QUOTE_NONE)
    glove = {}

    for i in df.index:
        glove[df[0][i]] = list(df.iloc[i, 1:])

    return glove


def vectorize(text: list, embeddings: dict, weights=None) -> list:
    vectorized = [0] * NO_GLOVE_DIMENSIONS

    for i, word in enumerate(text):
        if word in embeddings:
            word_embedding = embeddings[word]

            for j in range(NO_GLOVE_DIMENSIONS):
                if weights is not None:
                    vectorized[j] += (weights[i] * word_embedding[j])

                else:
                    vectorized[j] += word_embedding[j]

    return vectorized


def read_data(file: str, preprocessing: str, weights=None) -> (list, list):
    print("Preprocessing data")
    assert preprocessing in PREPROCESSING_METHODS, f"Preprocessing method \"{preprocessing}\" is not supported!"

    start_time = datetime.now()
    df = pd.read_csv(file, quoting=csv.QUOTE_NONE)

    if preprocessing == "pretrained":
        pass

    X = []
    y = []
    embeddings = get_glove_embeddings()

    for i in df.index:
        sample = tokenize(df["text"][i])

        if weights is not None:
            sample = vectorize(sample, embeddings, weights[i])

        else:
            sample = vectorize(sample, embeddings)

        X.append(sample)
        y.append(df["label"][i])

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return X, y


def write_preds(file: str, preds: list):
    print("Writing predictions to file")

    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])

        writer.writeheader()

        for i, pred in enumerate(preds):
            writer.writerow({"id": i, "label": pred})


def read_prob_weights_cached(file: str):
    df = pd.read_csv(file, quoting=csv.QUOTE_NONE)
    prob_per_word = {}

    for i in df.index:
        prob_per_word[df["word"][i]] = df["weight"][i]

    return prob_per_word
