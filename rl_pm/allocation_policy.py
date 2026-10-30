import torch
from torch import nn
from torch.distributions import Beta, Dirichlet

from .train import ActorCritic


class CashAndStocks:
    def __init__(self, stock_scores, cash_scores):
        self.stock = Dirichlet(torch.nn.functional.softplus(stock_scores) + .2)
        cash_concentration = torch.nn.functional.softplus(cash_scores) + 1.0
        self.cash = Beta(cash_concentration[0], cash_concentration[1])

    @property
    def mean(self):
        cash = self.cash.mean
        return torch.cat([self.stock.mean * (1 - cash), cash.reshape(1)])

    def sample(self):
        cash = self.cash.sample().clamp(1e-6, 1 - 1e-6)
        return torch.cat([self.stock.sample() * (1 - cash), cash.reshape(1)])

    def log_prob(self, action):
        cash = action[-1].clamp(1e-6, 1 - 1e-6)
        stock = (action[:-1] / (1 - cash)).clamp_min(1e-12)
        stock = stock / stock.sum()
        return self.cash.log_prob(cash) + self.stock.log_prob(stock)


class AllocationPolicy(ActorCritic):
    def __init__(self, input_size, hidden):
        super().__init__(input_size, hidden)
        self.cash_head = nn.Linear(hidden + 1, 2)

    def forward(self, observation):
        state = observation["state"]
        indices = observation["indices"]
        encoded = self.encoder(state)
        relevant = (state[:, -2] > 0) | (state[:, -3] > 0)
        pooled = encoded[relevant].mean(dim=0) if relevant.any() else encoded.new_zeros(encoded.shape[-1])
        context = torch.cat([pooled, observation["cash"].reshape(1)])
        value = self.value_head(context).squeeze()
        if len(indices) == 0:
            return None, value
        return CashAndStocks(self.stock_head(encoded[indices]).squeeze(-1), self.cash_head(context)), value
