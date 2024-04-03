import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import read_data, write_preds, get_rtf_igm_weights, read_prob_weights_cached, get_rtf_igm_test_weights

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model():
    def __init__(self, mode="debug", preprocessing="glove"):
        self.mode = mode
        weights = None
        prob_per_word = None

        if preprocessing == "glove_rtf_igm" and not os.path.isfile("data/cache_prob_weights.csv"):
            weights, prob_per_word = get_rtf_igm_weights("data/train_2024.csv", cache="data/cache_prob_weights.csv")

        elif preprocessing == "glove_rtf_igm":
            print("Cache file found for rtf-igm weights")

            prob_per_word = read_prob_weights_cached("data/cache_prob_weights.csv")
            weights, prob_per_word = get_rtf_igm_weights("data/train_2024.csv", prob_per_word=prob_per_word)

        if self.mode == "debug":
            X, y = read_data("data/train_2024.csv", preprocessing, weights=weights)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1)

        elif self.mode == "release":
            self.X_train, self.y_train = read_data("data/train_2024.csv", preprocessing, weights=weights)

            if preprocessing == "glove_rtf_igm":
                weights = get_rtf_igm_test_weights("data/test_2024.csv", prob_per_word)

            self.X_test, self.y_test = read_data("data/test_2024.csv", preprocessing, weights=weights)

        else:
            raise Exception(f"Mode \"{self.mode}\" is not supported!")
        


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