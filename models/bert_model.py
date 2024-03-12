#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.nn import functional as F
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

MODEL_NAME = "bert-base-uncased"

occurrences = {}
occurrences['0'] = {}
occurrences['1'] = {}

class BertModel:

    def __init__(self, model_name: str, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
        self.freeze_bert_layers()

    def freeze_bert_layers(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
            
            
    def preprocess(self, text):
        text = word_tokenize(text)
        text = map(lambda sample: sample.lower(), text)

        # stopword removal
        stop_words = set(stopwords.words("english"))
        text = list(filter(lambda sample: sample not in stop_words, text))
        return ' '.join(text)      
     
    
    def count(self, sentences, sentences_labels):
        for i in range(len(sentences)):
            text = word_tokenize(sentences[i])
            for word in text:
                if sentences_labels[i] == 0:
                    occurrences['0'][word] = occurrences['0'].get(word, 0) + 1
                else:
                    occurrences['1'][word] = occurrences['1'].get(word, 0) + 1
    
                    
    
    def reduce(self, sentencelist, words_to_pick):
        all_sentences = []
        for sentence in sentencelist:
            top_words = []
            text = word_tokenize(sentence)
            scores = {}
            for word in text:
                if word not in scores:
                    word_total = occurrences['1'].get(word, 0) + occurrences['0'].get(word, 0)
                    if word_total > 0:
                        scores[word] = abs(occurrences['1'].get(word, 0)/word_total - occurrences['0'].get(word, 0)/word_total)
                    else:
                        scores[word] = 0
            best_words = sorted(scores, key=scores.get, reverse=True)[:words_to_pick]
            added_words = 0
            for word in text: #get the original word order in sentence
                if word in best_words and added_words < words_to_pick:
                    top_words.append(word)
                    added_words += 1
            all_sentences.append(' '.join(top_words))
        #print(all_sentences)
        return all_sentences

                
    def tokenize(self, dataset: Dataset, maximum: int, training_bool: bool):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def help_function(sentences):
            sentences["text"] = [self.preprocess(text) for text in sentences["text"]]
            if training_bool:
                self.count(sentences["text"], sentences["label"])
            new_sentences = self.reduce(sentences["text"], 30)
            return tokenizer(new_sentences, padding='max_length', truncation=True, max_length=maximum)
            
        return dataset.map(help_function, batched = True)
    

    def run(self, train_data: Dataset, test_data: Dataset):
        trainer = Trainer(
            model = self.model,
            args = TrainingArguments(output_dir = "trainer", per_device_train_batch_size=self.batch_size,
                                     num_train_epochs=3, learning_rate=2e-3),
            train_dataset = train_data)

        trainer.train()
        output = trainer.predict(test_data)
        return output.predictions, output.label_ids
    

model = BertModel(MODEL_NAME)

train_df = pd.read_csv("snlp_toxicity_detection/data/train_2024.csv")
train_df = train_df[["text", "label"]]

smaller_train_df = Dataset.from_pandas(train_df[:48000])
smaller_test_df = Dataset.from_pandas(train_df[48000:60000])

tokenized_train = model.tokenize(smaller_train_df, maximum = 30, training_bool=True)
tokenized_test = model.tokenize(smaller_test_df, maximum = 30, training_bool=False)

preds, labels = model.run(tokenized_train, tokenized_test)

# Converting logits to probabilities
probs = F.softmax(torch.tensor(preds), dim=1)
#print("probs: ")
#print(probs)
predicted_class = torch.argmax(probs, dim=1)
print("preds: ")
print(predicted_class.numpy())
print("true labels: ")
print(labels)

print("F1-score is: ")
print(f1_score(labels, predicted_class.numpy()))

print("Accuracy is: ")
print(accuracy_score(labels, predicted_class.numpy()))