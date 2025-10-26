from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import warnings

from hpo_rl.trainers.base import BaseTrainer
from hpo_rl.core.factory import build_optimizer, get_criterion_instance


class TorchTrainer(BaseTrainer[nn.Module, DataLoader]):

    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
        "optimizer": {"type": str, "default": "Adam"},
        "learning_rate": {"type": float, "default": 0.001},
        "epochs": {"type": int, "default": 1},
        "criterion": {"type": str, "default": "CrossEntropyLoss"},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, model: nn.Module,
              train_data: DataLoader, val_data: Optional[DataLoader] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        model.to(self.device)

        optimizer = build_optimizer(model, self._hparams)
        criterion = get_criterion_instance(self._hparams['criterion'])

        history = {
            "train_loss_history": [],
            "val_loss_history": [],
        }

        num_epochs = self._hparams['epochs']
        epoch_iterator = tqdm(
            range(num_epochs),
            desc="Training Progress",
            leave=True
        )

        for epoch in epoch_iterator:
            model.train()
            total_train_loss = 0.0
            num_train_batches = len(train_data)

            if num_train_batches == 0:
                warnings.warn("Обучающая выборка пуста. Пропуск обучения.", UserWarning)
                continue

            for data, target in train_data:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / num_train_batches
            history["train_loss_history"].append(avg_train_loss)

            avg_val_loss = None
            if val_data is not None:
                model.eval()
                total_val_loss = 0.0
                num_val_batches = len(val_data)

                if num_val_batches > 0:
                    with torch.no_grad():
                        for data, target in val_data:
                            data, target = data.to(self.device), target.to(self.device)
                            output = model(data)
                            loss = criterion(output, target)
                            total_val_loss += loss.item()

                    avg_val_loss = total_val_loss / num_val_batches
                    history["val_loss_history"].append(avg_val_loss)

            postfix_stats = {"train_loss": f"{avg_train_loss:.4f}"}
            if avg_val_loss is not None:
                postfix_stats["val_loss"] = f"{avg_val_loss:.4f}"
            epoch_iterator.set_postfix(postfix_stats)

        return model, history
