from hpo_rl.experiments.run_experiment import run_experiment
from hpo_rl.experiments.run_experiment import run_n_experiments
from hpo_rl.models.simple_cnn import SimpleCNN
from hpo_rl.trainers.torch_trainer import TorchTrainer
from hpo_rl.data_processing.processors import pytorch_mnist_processor
from hpo_rl.nets.masked_net import MaskedNet
from torch.optim import Adam
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.utils.net.discrete import DiscreteCritic
from tianshou.utils.net.continuous import ContinuousActorProbabilistic
from tianshou.utils.net.continuous import ContinuousCritic
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.common import Recurrent
from tianshou.algorithm.modelfree.sac import SACPolicy
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
    config_ppo = {
    "full_args": {
        "algorithm":
        {
            "name": "ppo",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            "actor": DiscreteActor,
            "critic": DiscreteCritic, 
            # "layer_num": 3,
            # "hidden_layer_size": 64,
            "hidden_sizes": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 100,
            "epoch_num_steps": 100,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": ProbabilisticActorPolicy,
            "dist_fn": torch.distributions.Categorical,
            "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 1,
        "num_test_envs": 1,
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        # "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 2, 5],
        # "action_type": "continuous"
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 10
    }
    }
    config_rainbow = {
    "full_args": {
        "algorithm":
        {
            "name": "rainbow",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            # "actor": DiscreteActor,
            # "critic": DiscreteCritic, 
            "hidden_sizes": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 10,
            "epoch_num_steps": 1000,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            # "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": C51Policy,
            # "dist_fn": torch.distributions.Categorical,
            # "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        # "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }
    config_sac = {
    "full_args": {
        "algorithm":
        {
            "name": "sac",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            "actor": ContinuousActorProbabilistic,
            "critic": ContinuousCritic, 
            "hidden_sizes": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 100,
            "epoch_num_steps": 100,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            # "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": SACPolicy,
            # "dist_fn": torch.distributions.Categorical,
            "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        # "num_bins": 300,
        "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }
    config_dqn = {
    "full_args": {
        "algorithm":
        {
            "name": "dqn",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },
        "buffer":
        {
            "total_size": 10000,
            "buffer_num": 10,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            # "actor": DiscreteActor,
            # "critic": DiscreteCritic, 
            "hidden_sizes": [64, 64],
            "net": MaskedNet
        },
        "trainer":
        {
            "max_epochs": 100,
            "epoch_num_steps": 100,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            # "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": DiscreteQLearningPolicy,
            # "dist_fn": torch.distributions.Categorical,
            # "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "new_cycle_move_pipeline",
        "num_bins": 300,
        # "action_type": "continuous",
        "max_steps": 100,
        # "reward_mode": "per_step",
        "step_sizes": [1, 2, 5],
        # "use_history": True
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }

    config_reinforce = {
    "full_args": {
        "algorithm":
        {
            "name": "sac",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            "actor": DiscreteActor,
            "critic": DiscreteCritic, 
            "hidden_states": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 10,
            "epoch_num_steps": 1000,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            # "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": ProbabilisticActorPolicy,
            "dist_fn": torch.distributions.Categorical,
            "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        # "num_bins": 300,
        "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }

    config_TPE = {
        "backend": {
        "name": "objective",
        "num_epochs": 1,
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
        "full_args": {
        "algorithm": {
            "name": "TPE",
            "N_init": 20,
            "N_s": 100,
            "budget": 20,
            "separation_value": 0.2
        }
        }
    }

    config_SimpleGA = {
        "backend": {
            "name": "function",
            "function": "rastrigin",
            "dimensions": 2
        },
        "algorithm": {
            "name": "SimpleGA",
            "N_pop": 20,
            "budget": 100,
            "mutation_prob": 0.1,
            "crossover_prob": 0.8,
            "tournament_size": 3,
            "elitism": True
        }
    }

    config_CMA_ES = {
        "backend": {
            "name": "function",
            "function": "rastrigin",
            "dimensions": 2
        },
        "algorithm": {
            "name": "CMA_ES",
            "N_pop": 10,
            "budget": 100,
            "initial_step_size": 0.5
        }
    }

    config_real = {
        "full_args": {
        "algorithm":
        {
            "name": "ppo",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            "actor": DiscreteActor,
            "critic": DiscreteCritic, 
            # "layer_num": 3,
            # "hidden_layer_size": 64,
            "hidden_sizes": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 1,
            "epoch_num_steps": 4,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": ProbabilisticActorPolicy,
            "dist_fn": torch.distributions.Categorical,
            "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 2,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "real",
        "model": SimpleCNN,
        "trainer": TorchTrainer,
        "data_processor": pytorch_mnist_processor,
        "hp_space": {
            "n_params": {
                "refers_to": "model",
                "type": "int",
                "min": 1,
                "max": 512
            },
            "learning_rate":{
                "type": "float",
                "min": 0,
                "max": 0.1
            },
            "optimizer": {
                "refers_to": "train_loop",
                "type": "categorical",
                "values": ["SGD", "Adam"],
                "dependencies": ["learning_rate"]
            },
            "criterion": {
                "refers_to": "train_loop",
                "type": "categorical",
                "values": ["CrossEntropyLoss"]
            },
            "learning_rate": {
                "refers_to": "optimizer",
                "type": "float",
                "min": 0,
                "max": 1
            }
        }
    }
    }
    run_n_experiments(config_dqn, 3)
