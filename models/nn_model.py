import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model import Model
from models.alexnet import AlexNet
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from utils import pad_sentences


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(np.array(self.X[idx]), dtype=torch.float32)
        y = torch.tensor(np.array(self.y[idx]), dtype=torch.float32)

        return X, y


class NNModel(Model):
    def __init__(self, mode="debug", preprocessing="glove_rtf_igm"):
        super(NNModel, self).__init__(mode, preprocessing, separate_word_embeddings=True)

        self.X_train = pad_sentences(self.X_train)
        self.X_test = pad_sentences(self.X_test)

        self.train_data = CustomDataset(self.X_train, self.y_train)
        self.test_data = CustomDataset(self.X_test, self.y_test)

    def run(self):
        model = AlexNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loader = DataLoader(dataset=self.train_data, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=self.test_data, batch_size=self.test_data.__len__(), shuffle=False)
        loss_fun = nn.BCEWithLogitsLoss()

        print("Training")

        for epoch in range(3):
            print(f"Epoch {epoch}")
            model.train()

            for (sentences, labels) in train_loader:
                optimizer.zero_grad()
                outputs = model.forward(sentences)
                loss = loss_fun(outputs, labels)

                loss.backward()
                optimizer.step()

        print("Testing")

        if self.mode == "debug":
            model.eval()

            with torch.no_grad():
                sentences, labels = next(iter(test_loader))
                y = model(sentences)

                print(f1_score([sample > 0 for sample in y], labels))

        elif self.mode == "release":
            pass

        else:
            raise Exception(f"Mode \"{self.mode}\" is not supported!")
