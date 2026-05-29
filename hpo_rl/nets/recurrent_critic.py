from torch import nn

class RecurrentCritic(nn.Module):
    """Критик V(s): preprocess_net → линейная голова на 1.

    Args:
        preprocess_net: рекуррентный или плоский backbone с ``output_dim``.
    """

    def __init__(self, preprocess_net):
        """Инициализирует RecurrentCritic.

        Args:
            preprocess_net: сеть признаков.
        """
        super().__init__()
        self.preprocess = preprocess_net

        self.last = nn.Linear(preprocess_net.output_dim, 1)

    def forward(self, obs, state=None, info=None):

        x, hidden = self.preprocess(obs, state=state, info=info)

        v = self.last(x)


        v = v.squeeze(-1)
        return v
