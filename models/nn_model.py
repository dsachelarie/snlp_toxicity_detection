import torch
import torch.optim as optim

from models.model import Model
from models.alexnet import AlexNet
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        X = torch.tensor(self.X_train[idx], dtype=torch.float32)
        y = torch.tensor(self.y_train[idx], dtype=torch.float32)

        return X, y


class NNModel(Model):
    def __init__(self, mode="debug", preprocessing="glove_rtf_igm"):
        super(NNModel, self).__init__(mode, preprocessing, separate_word_embeddings=True)

        self.train_data = CustomDataset(self.X_train, self.y_train)
        self.test_data = CustomDataset(self.X_test, self.y_test)

    def run(self):
        model = AlexNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loader = DataLoader(dataset=self.train_data, batch_size=256, shuffle=True)

        for epoch in range(3):
            model.train()

            # TODO
