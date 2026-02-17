from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import warnings

from collections import defaultdict

from hpo_rl.trainers.base import BaseTrainer
from hpo_rl.core.factory import build_optimizer, get_criterion_instance


class TorchTrainer(BaseTrainer[nn.Module, DataLoader]):


    def __init__(self, config):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # получать гиперпараметры из конфига добавить
    def train(self, config, model, train_data, val_data) -> Tuple[nn.Module, Dict[str, Any]]:
        optimizer_name = config.get("optimizer", "default")
        criterion_name = config.get("criterion", "default")

        if optimizer_name == "default":
            optimizer_name = "Adam"
            warnings.warn("Optimizer is not specified in config, using default: optimizer=Adam")
        
        if criterion_name == "default":
            criterion_name = "CrossEntropyLoss"
            warnings.warn("Criterion is not specified in config, using default: criterion=CrossEntropyLoss")
        
        defdict_config = defaultdict(dict)
        
        # print("config:", config)

        for key, meta in self.hp_space.items():
            target = meta.get("refers_to")
            if key in config:
                # Записываем в группу (например, 'train_loop') значение под его именем
                defdict_config[target][key] = config[key]

        # Превращаем обратно в обычный словарь для вывода
        parsed_config = dict(defdict_config)

        model = model(**(parsed_config.get("model")))
        model.to(self.device)

        optimizer = build_optimizer(model, {"optimizer": optimizer_name})
        criterion = get_criterion_instance(criterion_name)
        # добавить scheduler

        history = {
            "train_loss_history": [],
            "val_loss_history": [],
        }

        num_epochs = parsed_config.get("train_loop", 2).get('epochs', 2)
        epoch_iterator = tqdm(
            range(num_epochs),
            desc="Training Progress",
            position=1,
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


