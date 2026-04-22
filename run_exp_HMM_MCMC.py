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

config_HMM = {
    "full_args": {
        "algorithm": {
            "name": "HMM_MCMC",
            "budget": 800,
            "n_init": 5,           # было 5 — 20 Sobol-точек даёт достаточное покрытие 10D
            "n_chains": 1,
            "orchestrate_every": 1000, # было 1000 (> budget) — оркестратор не срабатывал никогда
            "T_mcmc": 0.01,
            "sigma_fraction": 0.0055,
            "wide_sigma_fraction": 0.5, # было 0.5 — чуть менее агрессивный EXPLORE
            "temperature": 0.3,
            "hmm_window": 4,
            "hmm_obs_epsilon": 1e-8,
            "hmm_lambda_noise": 0.01,
            "clone_noise": 0.05,
            "burnin_fraction": 0.0,
            "p_cat_step": 0.0,
            "kde_tau": 0.05,
            "anneal_T": True
        }
    },
    "backend": {
        "name": "function", "function": "schwefel", "dimensions": 10, "noise_std": 0
    }
}

config_TPE = {
    "full_args": {
        "algorithm": {
            "name": "TPE",
            "N_init": 5,
            "N_s": 20,
            "budget": 800,
            "separation_value": 0.2
        }
    },
    "backend": {
        "name": "objective",
        "objective_function": objective_function,
        "hp_space": {
            "lr": {
                "type": "float", 
                "values": [1e-6,1e-2]
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
    },
}

if __name__ == "__main__":
    run_n_experiments(config_HMM, n_experiments=3, inference_only=False)
