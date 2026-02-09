from hpo_rl.experiments.run_experiment import run_experiment
from hpo_rl.experiments.run_experiment import run_n_experiments
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100, n_params=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, n_params) 
        self.fc2 = nn.Linear(n_params, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 64 * 8 * 8) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def objective_function(config, dict_config):
    param_values = {}
    for name in dict_config.keys():
        param_values[name] = config[name]

    n_params = param_values["n_params"]
    lr = param_values["lr"]
    batch_size = int(param_values["batch_size"])
    optimizer_name = param_values["optimizer"]

    transform = transforms.ToTensor()
    
    try:
        dataset = datasets.CIFAR100(root='./tmp_data', train=True, download=True, transform=transform)
    except:
        dataset = datasets.CIFAR100(root='./tmp_data', train=True, download=False, transform=transform)
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes=100, n_params=n_params) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    
    sub_bar = tqdm(total=int(2),desc="Model training", position=1, leave=False)
    
    for epoch in range(int(2)):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        sub_bar.update(1)
    
    sub_bar.close()

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / len(val_dataset)

    print(f"Config: {param_values}, ValLoss: {avg_val_loss:.4f}, ValAcc: {val_accuracy:.4f}")

    return avg_val_loss


if __name__ == "__main__":

    config = {
    "algorithm": {
        "name": "PPO", 
        "verbose": 1,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "total_timesteps": 5,
        "inference_timesteps": 5,
        "n_steps": 5,
        "batch_size": 5,
        "policy": "MultiInputPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 6,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "objective",
        "num_epochs": 1,
        "objective_function": objective_function,
        "hp_space": {
            "lr": {
                "type": "float", 
                "min": 1e-6,
                "max": 1e-2
            },
            "batch_size": {
                "type": "categorical", 
                "values": [32, 64, 128]
            },
            "optimizer": {
                "type": "categorical", 
                "values": ["Adam", "SGD"]
            },
            "n_params": {
                "type": "categorical", 
                "values": [16, 32, 64] 
            }
        }
    }
    }
    config_BOHB = {
    "algorithm": {
        "name": "BOHB",
        "R": 2,
        "nu": 2,
    },
    "backend": {
        "name": "objective",
        "num_epochs": 1,
        "objective_function": objective_function,
        "hp_space": {
            "lr": {
                "type": "float", 
                "min": 1e-6,
                "max": 1e-2
            },
            "batch_size": {
                "type": "categorical", 
                "values": [32, 64, 128]
            },
            "optimizer": {
                "type": "categorical", 
                "values": ["Adam", "SGD"]
            },
            "n_params": {
                "type": "categorical", 
                "values": [16, 32, 64] 
            }
        }
    }
    }
    
    run_experiment(config_BOHB)