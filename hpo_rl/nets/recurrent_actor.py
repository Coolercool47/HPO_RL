from tianshou.utils.net.discrete import DiscreteActor
import torch
from tianshou.data import Batch

class MaskedRecurrentDiscreteActor(DiscreteActor):
    """Дискретный актор для рекуррентных сетей с action masking.

        Args:
            preprocess_net: рекуррентный backbone
            action_shape: форма действий
            hidden_sizes: не используется (голова — один Linear)

        Note:
            Стандартный DiscreteActor делает flatten(1), что ломает 3D вход
            ``[batch, seq_len, hidden_dim]`` от RNN. Этот класс вызывает
            preprocess_net напрямую и накладывает mask на логиты.
    """
    def __init__(self, preprocess_net, action_shape, hidden_sizes=()):
        """Инициализирует MaskedRecurrentDiscreteActor.

        Args:
            preprocess_net: рекуррентный backbone.
            action_shape: форма действий.
            hidden_sizes: не используется (голова — один Linear).
        """
        super().__init__(
            preprocess_net=preprocess_net,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            softmax_output=False
        )

    def forward(self, obs, state=None, info=None):

        x, hidden = self.preprocess(obs, state=state, info=info)


        is_3d = x.dim() == 3
        if is_3d:
            batch_size, seq_len, feat_dim = x.shape
            x_flat = x.reshape(batch_size * seq_len, feat_dim)
            logits = self.last(x_flat)
            logits = logits.reshape(batch_size, seq_len, -1)
        else:
            logits = self.last(x)


        mask = None
        if isinstance(obs, (dict, Batch)) and "mask" in obs:
            mask = obs["mask"]
        elif hasattr(obs, "mask"):
            mask = obs.mask

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)


            if mask.dim() < logits.dim():
                mask = mask.unsqueeze(1)

            logits = logits.masked_fill(~mask, -1e8)

        return logits, hidden
