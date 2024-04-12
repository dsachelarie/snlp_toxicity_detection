import torch
import torch.nn as nn


NO_GLOVE_DIMENSIONS = 300
MAX_SENTENCE_LENGTH = 200


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=(11, NO_GLOVE_DIMENSIONS)),
          nn.ELU(),
          nn.MaxPool2d(kernel_size=(MAX_SENTENCE_LENGTH - 10, 1))
        )

        self.conv2 = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=(5, NO_GLOVE_DIMENSIONS)),
          nn.ELU(),
          nn.MaxPool2d(kernel_size=(MAX_SENTENCE_LENGTH - 4, 1))
        )

        self.conv3 = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=(3, NO_GLOVE_DIMENSIONS)),
          nn.ELU(),
          nn.MaxPool2d(kernel_size=(MAX_SENTENCE_LENGTH - 2, 1))
        )

        self.dropout1 = nn.Dropout(0.4)

        self.fc = nn.Sequential(
            nn.Linear(192, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, MAX_SENTENCE_LENGTH, NO_GLOVE_DIMENSIONS): Input embeddings.

        Returns:
          y of shape (batch_size, 1): Output.
        """

        x = x.unsqueeze(1)
        x1 = self.conv1(x).squeeze(3)
        x2 = self.conv2(x).squeeze(3)
        x3 = self.conv3(x).squeeze(3)

        y = torch.flatten(self.dropout1(torch.cat((x1, x2, x3), dim=2)), start_dim=1, end_dim=2)
        y = self.dropout2(self.fc(y))

        return y.squeeze(1)
