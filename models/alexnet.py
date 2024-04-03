import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.dropout1 = nn.Dropout2d(0.4)

        self.conv1 = nn.Sequential(
          nn.Conv2d(300, 64, kernel_size=(11, 11)),
          nn.ELU(),
          nn.MaxPool2d(kernel_size=11)
        )

        self.conv2 = nn.Sequential(
          nn.Conv2d(300, 64, kernel_size=(5, 5)),
          nn.ELU(),
          nn.MaxPool2d(kernel_size=5)
        )

        self.conv3 = nn.Sequential(
          nn.Conv2d(300, 64, kernel_size=(3, 3)),
          nn.ELU(),
          nn.MaxPool2d(kernel_size=3)
        )

        self.dropout2 = nn.Dropout(0.4)

        self.fc = nn.Sequential(
            nn.Linear(192, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 200, 300): Input embeddings.

        Returns:
          y of shape (batch_size, 1): Output.
        """

        x = self.dropout1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        y = torch.flatten(self.dropout2(torch.cat((x1, x2, x3), dim=1)))

        print(y.shape)

        y = self.fc(y)

        return y
