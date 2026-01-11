import torch.nn as nn
import torch.nn.functional as F
from hpo_rl.models.base import BaseModel


class SimpleCNN(BaseModel):
    """
    A small CNN model for experiments and demonstration.
    Provides a simple convolutional neural network architecture for image classification.
    """
    HYPERPARAMETERS = {
        "num_classes": {"type": int, "default": 10},  # Number of output classes
        "n_params": {"type": int, "default": 128}     # Size of the penultimate layer
    }  # Can be extended with NAS parameters like number of conv layers

    def __init__(self, num_classes, n_params):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, n_params)
        self.fc2 = nn.Linear(n_params, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
