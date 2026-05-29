from torch import nn

class RecurrentCritic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net

        self.last = nn.Linear(preprocess_net.output_dim, 1)

    def forward(self, obs, state=None, info=None):

        x, hidden = self.preprocess(obs, state=state, info=info)

        v = self.last(x)


        v = v.squeeze(-1)
        return v
