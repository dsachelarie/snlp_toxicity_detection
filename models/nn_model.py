import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module, Model):
    def __init__(self, mode="debug", preprocessing="glove_rtf_igm"):
        super(ConvNet, self).__init__(mode, preprocessing)
        
        self.conv1 = nn.Sequential(
          
          nn.Conv1d(300, 200, kernel_size=11)
          nn.ELU()
          nn.MaxPool1d(kernel_size=11)
        )

        self.conv2 = nn.Sequential(
          
          nn.Conv1d(300, 200, kernel_size=5)
          nn.ELU()
          nn.MaxPool1d(kernel_size=5)
        )

        self.conv3 = nn.Sequential(
          
          nn.Conv1d(300, 200, kernel_size=3)
          nn.ELU()
          nn.MaxPool1d(kernel_size=3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3, 256)
            nn.Dropout(0.2)
            nn.Linear(256, 1)
        )
        

    def forward(self, x, verbose=False):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
          verbose: True if you want to print the shapes of the intermediate variables.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        # YOUR CODE HERE

        
        return y