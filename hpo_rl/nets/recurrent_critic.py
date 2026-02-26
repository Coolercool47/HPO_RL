from torch import nn

class RecurrentCritic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        # Добавляем финальный слой для оценки ценности состояния (Value)
        self.last = nn.Linear(preprocess_net.output_dim, 1)

    def forward(self, obs, state=None, info=None):
        # 1. Прогоняем через нашу RNN
        x, hidden = self.preprocess(obs, state=state, info=info)
        # 2. Получаем оценку
        v = self.last(x)
        # 3. КРИТИЧЕСКИ ВАЖНО: Убираем последнюю размерность!
        # [Batch, Seq_len, 1] -> [Batch, Seq_len]
        v = v.squeeze(-1)
        return v