from hpo_rl.models.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(BaseModel):
    """
    Модель маленькой CNN для экспериментов и примера
    """
    # Контракт для гиперпараметров модели
    HYPERPARAMETERS = {
        "num_classes": {"type": int, "default": 10},  # сильно связано с данными, и вообще говоря не то чтобы прям подбирается
        "n_params": {"type": int, "default": 128}
    }  # Теоретически можно добавить шизоNAS с числом сверточных слоев

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
