from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import csv
import math

from pandas import DataFrame

nltk.download("stopwords")

GLOVE_PATH = "data/glove.6B.300d.txt"
NO_GLOVE_DIMENSIONS = 300
MAX_SENTENCE_LENGTH = 300
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


def stem(text: list) -> list:
    stemmer = PorterStemmer()
    stemmed = []

    for word in text:
        stemmed.append(stemmer.stem(word))

    return stemmed


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


def get_igm_weights(data: DataFrame) -> dict:
    print("Calculating igm weights")
    start_time = datetime.now()

    stem_counts = {0: {}, 1: {}}

    for i in data.index:
        sample = data["stemmed_text"][i]

        for word in sample:
            if word in stem_counts[data["label"][i]]:
                stem_counts[data["label"][i]][word] += 1

            else:
                stem_counts[data["label"][i]][word] = 1

    no_words_per_label = [sum(stem_counts[0].values()), sum(stem_counts[1].values())]
    prob_per_word = {}

    for i in data.index:
        sample = data["stemmed_text"][i]

        for word in sample:
            if word not in prob_per_word:
                prob_per_word[word] = igm(word, stem_counts, no_words_per_label)

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return prob_per_word


def get_glove_embeddings() -> dict:
    print("Fetching GloVe embeddings")
    start_time = datetime.now()

    df = pd.read_csv("data/glove.6B.300d.txt", header=None, sep=" ", quoting=csv.QUOTE_NONE)
    glove = {}

    for i in df.index:
        glove[df[0][i]] = list(df.iloc[i, 1:])

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return glove


def vectorize(text: list, stemmed_text: list, embeddings: dict, igm_weights=None, separate_word_embeddings=False) -> list:
    if separate_word_embeddings:
        vectorized = list(np.zeros((len(text), NO_GLOVE_DIMENSIONS)))
    else:
        vectorized = [0] * NO_GLOVE_DIMENSIONS

    for i, word in enumerate(text):
        if word in embeddings:
            word_embedding = embeddings[word]
            stemmed_word = stemmed_text[i]

            for j in range(NO_GLOVE_DIMENSIONS):
                if separate_word_embeddings:
                    if igm_weights is not None and stemmed_word in igm_weights:
                        vectorized[i][j] = math.sqrt(stemmed_text.count(stemmed_word) / len(stemmed_text)) * \
                                           igm_weights[stemmed_word] * word_embedding[j]

                    # When we have no "igm" information, we consider the word to be occurring with equal frequency in both classes
                    elif igm_weights is not None:
                        vectorized[i][j] = math.sqrt(stemmed_text.count(stemmed_word) / len(stemmed_text)) * \
                                           (1 + IGM_LAMBDA) * word_embedding[j]

                    else:
                        vectorized[i][j] = word_embedding[j]

                else:
                    if igm_weights is not None and stemmed_word in igm_weights:
                        vectorized[j] += (math.sqrt(stemmed_text.count(stemmed_word) / len(stemmed_text)) *
                                          igm_weights[stemmed_word] * word_embedding[j])

                    # When we have no "igm" information, we consider the word to be occurring with equal frequency in both classes
                    elif igm_weights is not None:
                        vectorized[j] += math.sqrt(stemmed_text.count(stemmed_word) / len(stemmed_text)) * \
                                         (1 + IGM_LAMBDA) * word_embedding[j]

                    else:
                        vectorized[j] += word_embedding[j]

    return vectorized


def read_data(file: str) -> DataFrame:
    print("Reading data")
    start_time = datetime.now()

    data = pd.read_csv(file, quoting=csv.QUOTE_NONE)
    data["text"] = data["text"].apply(tokenize)
    data["stemmed_text"] = data["text"].apply(stem)

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return data


def add_embeddings(data: DataFrame, embeddings: dict, igm_weights=None, separate_word_embeddings=False) -> DataFrame:
    print("Adding embeddings")
    start_time = datetime.now()

    data["embedding"] = data.apply(lambda sample: vectorize(sample["text"], sample["stemmed_text"],
                                                            embeddings, igm_weights=igm_weights,
                                                            separate_word_embeddings=separate_word_embeddings), axis=1)

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return data


def balance_dataset(data: DataFrame) -> DataFrame:
    print("Balancing dataset")
    start_time = datetime.now()

    data_0 = data[data["label"] == 0]
    data_1 = data[data["label"] == 1]
    data_0_sampled = data_0.sample(n=data_1.shape[0], replace=False)
    balanced_data = pd.concat([data_0_sampled, data_1])
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

    print(f"Completed in {round((datetime.now() - start_time).total_seconds())} seconds")

    return balanced_data


def get_embeddings_only(data: DataFrame) -> (list, list):
    return data["embedding"].tolist(), data["label"].tolist()


def write_preds(file: str, preds: list):
    print("Writing predictions to file")

    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])

        writer.writeheader()

        for i, pred in enumerate(preds):
            writer.writerow({"id": i, "label": pred})


def pad_sentences(data: list) -> list:
    empty_embedding = [0] * NO_GLOVE_DIMENSIONS

    for sample in data:
        for i in range(MAX_SENTENCE_LENGTH - len(sample)):
            sample.append(empty_embedding.copy())

    return data
