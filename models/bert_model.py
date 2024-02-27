from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score
import torch
import pandas as pd
import numpy as np

train_df = pd.read_csv("snlp_toxicity_detection/data/train_2024.csv")
train_df = train_df[["text", "label"]]

test_df = pd.read_csv("snlp_toxicity_detection/data/test_2024.csv")
test_df = test_df[["text"]]

train_ds = Dataset.from_pandas(train_df[:1000])
test_ds = Dataset.from_pandas(test_df)

MODEL_NAME = "bert-base-uncased"

class BertModel:

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
        self.freeze_bert_layers()

    def freeze_bert_layers(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def tokenize(self, dataset: Dataset, max_length: int):

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return dataset.map(lambda samples: tokenizer(samples["text"], padding = "max_length", truncation = True, max_length = max_length), batched = True)

    def run(self, train_data: Dataset, test_data: Dataset):
        trainer = Trainer(
            model = self.model,
            args = TrainingArguments(output_dir = "trainer", per_device_train_batch_size=self.batch_size),
            train_dataset = train_data
        )

        trainer.train()
        output = trainer.predict(test_data)

        return output.predictions, output.label_ids


model = BertModel(MODEL_NAME)

tokenized_train = model.tokenize(train_ds, max_length = 100)
tokenized_test = model.tokenize(test_ds, max_length = 100)

preds, labels = model.run(tokenized_train, tokenized_test)

print(f1_score(preds, labels))
