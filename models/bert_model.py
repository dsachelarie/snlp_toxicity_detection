#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, AdamW
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.nn import functional as F
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

MODEL_NAME = "bert-base-uncased"

occurrences_0 = {}
occurrences_1 = {}

class BertModel:

    def __init__(self, model_name: str, batch_size: int = 8, learning_rate = 5e-05):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2)
        #self.freeze_bert_layers()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate


    def freeze_bert_layers(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False


    def preprocess(self, text):
        text = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered = []
        for word in text:
            word = word.lower()
            if word not in stop_words:
                filtered.append(word)
        return ' '.join(filtered)


    def count(self, sentences, sentences_labels):
        for i in range(len(sentences)):
            text = word_tokenize(sentences[i])
            for word in text:
                if sentences_labels[i] == 0:
                    occurrences_0[word] = occurrences_0.get(word, 0) + 1
                else:
                    occurrences_1[word] = occurrences_1.get(word, 0) + 1


    def reduce(self, sentencelist, words_to_pick):
        all_sentences = []
        for sentence in sentencelist:
            top_words = []
            text = word_tokenize(sentence)
            scores = {}
            for word in text:
                if word not in scores:
                    word_total = occurrences_1.get(word, 0) + occurrences_0.get(word, 0)
                    if word_total > 0:
                        scores[word] = abs(occurrences_1.get(word, 0)/word_total - occurrences_0.get(word, 0)/word_total)
                    else:
                        scores[word] = 0
            best_words = sorted(scores, key=scores.get, reverse=True)[:words_to_pick]
            added_words = 0
            for word in text: #get the original word order in sentence
                if word in best_words and added_words < words_to_pick:
                    top_words.append(word)
                    added_words += 1
            all_sentences.append(' '.join(top_words))
        return all_sentences 


    def tokenize(self, dataset: Dataset, maximum: int, training_bool: bool):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        processed = []
        for sentence in dataset:
            processed_sentence = self.preprocess(sentence["text"])
            processed.append(processed_sentence)
        labels = dataset["label"]
        if training_bool:
            self.count(processed, labels)
        new_sentences = self.reduce(processed, maximum)
        tokenized = tokenizer(new_sentences, padding='max_length', truncation=True, max_length=maximum)
        data = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'token_type_ids': tokenized['token_type_ids'],
            'labels': labels
        }
        arrow_data = Dataset.from_dict(data)
        return arrow_data


    def run(self, train_data: Dataset, test_data: Dataset):
        trainer = Trainer(
            model = self.model,
            train_dataset = train_data)
        trainer.optimizer = self.optimizer
        trainer.train()
        output = trainer.predict(test_data)
        return output.predictions, output.label_ids


    
model = BertModel(MODEL_NAME)

train_df = pd.read_csv("data/train_2024.csv")
train_df = train_df[["text", "label"]]
test_df = pd.read_csv("data/dev_2024.csv")
test_df = test_df[["text", "label"]]

train_df_0 = train_df[train_df["label"] == 0]
train_df_1 = train_df[train_df["label"] == 1]
ones = train_df_1.shape[0]
train_df_0_sampled = train_df_0.sample(n=ones, replace=False)
new_train_df = pd.concat([train_df_0_sampled, train_df_1])
new_train_df = new_train_df.sample(frac=1).reset_index(drop=True)
new_train_df = Dataset.from_pandas(new_train_df) # :1250
test_df = Dataset.from_pandas(test_df) # :1000

tokenized_train = model.tokenize(new_train_df, maximum = 125, training_bool=True)
tokenized_test = model.tokenize(test_df, maximum = 125, training_bool=False)


preds, labels = model.run(tokenized_train, tokenized_test)
# Converting logits to probabilities
probs = F.softmax(torch.tensor(preds), dim=1)
predicted_class = torch.argmax(probs, dim=1)
print("preds: ")
print(predicted_class.numpy())
print("true labels: ")
print(labels)

print("F1-score is: ")
print(f1_score(labels, predicted_class.numpy()))
print("Accuracy is: ")
print(accuracy_score(labels, predicted_class.numpy()))
