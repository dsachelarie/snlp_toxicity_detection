from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import string
import csv
import math
nltk.download("stopwords")

GLOVE_PATH = "data/glove.6B.300d.txt"
NO_GLOVE_DIMENSIONS = 300
PREPROCESSING_METHODS = ["glove", "glove_tf_prob", "pretrained"]


def tokenize(text: str) -> list:
    # lowercasing and tokenization
    text = word_tokenize(text)
    text = map(lambda sample: sample.lower(), text)

    # stopword removal
    stop_words = set(stopwords.words("english"))
    text = list(filter(lambda sample: sample not in stop_words, text))

    # punctuation removal
    text = list(filter(lambda word: word not in string.punctuation, text))

    return text


def prob(label: int, word: str, counts: dict, words_which_samples: dict,
         no_words_per_label: list, no_samples_per_label: list) -> float:
    if word in counts[abs(label - 1)]:
        a_b = (counts[label][word] / no_words_per_label[label]) \
               / (counts[abs(label - 1)][word] / no_words_per_label[abs(label - 1)])

    else:
        a_b = (counts[label][word] / no_words_per_label[label]) \
               / (1 / no_words_per_label[abs(label - 1)])

    a_c = len(words_which_samples[label][word]) / no_samples_per_label[label]

    return math.log(1 + a_b * a_c)


def get_tf_prob_weights(file: str, cache=None) -> list:
    print("Calculating tf-prob weights")

    start_time = datetime.now()
    df = pd.read_csv(file, quoting=csv.QUOTE_NONE)
    stemmer = PorterStemmer()
    stems = []
    labels = []
    stem_counts = {0: {}, 1: {}}
    stem_which_samples = {0: {}, 1: {}}

    for i in df.index:
        sample = tokenize(df["text"][i])
        sample_stems = []

        for word in sample:
            stemmed_word = stemmer.stem(word)
            sample_stems.append(stemmed_word)

            if stemmed_word in stem_counts[df["label"][i]]:
                stem_counts[df["label"][i]][stemmed_word] += 1

                if i not in stem_which_samples[df["label"][i]][stemmed_word]:
                    stem_which_samples[df["label"][i]][stemmed_word].append(i)

            else:
                stem_counts[df["label"][i]][stemmed_word] = 1
                stem_which_samples[df["label"][i]][stemmed_word] = [i]

        stems.append(sample_stems)
        labels.append(df["label"][i])

    weights = []
    no_words_per_label = [sum(stem_counts[0].values()), sum(stem_counts[1].values())]

    for sample, label in zip(stems, labels):
        sample_weights = []

        for word in sample:
            sample_weights.append(sample.count(word) / len(sample) *
                                  prob(label, word, stem_counts, stem_which_samples,
                                       no_words_per_label, [labels.count(0), labels.count(1)]))

        weights.append(sample_weights)

    if cache is not None:
        with open(cache, "w", newline="") as f:
            for sample, label in zip(weights, labels):
                for weight in sample:
                    f.write(str(weight) + " ")

                f.write("," + str(label) + "\n")

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


def get_tf_prob_weights_cached(file: str):
    pass
