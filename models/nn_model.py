import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model import Model
from models.alexnet import AlexNet
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from utils import pad_trunc_sentences, write_preds


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

        # Padding and truncation will likely increase the size of the dataset. If the available memory
        # is not sufficient, consider using only a subset of the samples.
        self.X_train = pad_trunc_sentences(self.X_train)
        self.X_test = pad_trunc_sentences(self.X_test)

        self.train_data = CustomDataset(self.X_train, self.y_train)
        self.test_data = CustomDataset(self.X_test, self.y_test)

    def run(self):
        model = AlexNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loader = DataLoader(dataset=self.train_data, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=self.test_data, batch_size=self.test_data.__len__(), shuffle=False)
        loss_fun = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()

        print("Training")

        for epoch in range(3):
            print(f"Epoch {epoch}")
            model.train()
            loss_per_epoch = 0

            for (sentences, labels) in train_loader:
                optimizer.zero_grad()
                outputs = model.forward(sentences)
                loss = loss_fun(outputs, labels)
                loss_per_epoch += loss.item()

                loss.backward()
                optimizer.step()

            loss_per_epoch /= len(train_loader)
            print(loss_per_epoch)

        print("Testing")

        model.eval()

        with torch.no_grad():
            sentences, labels = next(iter(test_loader))
            y = sigmoid(model(sentences))

            if self.mode == "debug":
                print(f1_score([int(sample > 0.5) for sample in y], labels))

            elif self.mode == "release":
                write_preds("data/preds_alexnet.csv", [int(sample > 0.5) for sample in y])

            else:
                raise Exception(f"Mode \"{self.mode}\" is not supported!")
