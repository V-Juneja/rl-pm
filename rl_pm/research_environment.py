import numpy as np
import torch

from .train import Portfolio
from .cross_sectional import augment_features


class ResearchPortfolio(Portfolio):
    def observation(self):
        t = self.position
        active = self.market.eligible[t]
        held = self.weights > 0
        relevant = active | held
        values = augment_features(
            self.market.features[t], active, held,
        ).copy()
        if not np.isfinite(values[relevant]).all():
            raise ValueError("Nonfinite research features")
        values[~relevant] = self.normalizer.mean
        state = np.column_stack([
            self.normalizer.transform(values), self.weights.astype(np.float32),
            active.astype(np.float32), self.market.tradable[t].astype(np.float32),
        ])
        indices = np.flatnonzero(relevant & self.market.tradable[t])
        return {
            "state": torch.from_numpy(state), "indices": torch.from_numpy(indices),
            "cash": torch.tensor(self.cash, dtype=torch.float32),
        }

    def step(self, indices, action):
        t = self.position
        indices = np.asarray(indices, dtype=np.int64)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(indices) + 1,) or not np.isfinite(action).all():
            raise ValueError("Invalid research allocation shape or values")
        if (action < 0).any() or not np.isclose(action.sum(), 1, atol=1e-5):
            raise ValueError("Research allocation must be long-only and sum to one")
        allowed = np.flatnonzero((self.market.eligible[t] | (self.weights > 0)) & self.market.tradable[t])
        if not np.array_equal(indices, allowed):
            raise ValueError("Allocation indices do not match observable tradability")
        locked = (self.weights > 0) & ~self.market.tradable[t]
        fixed = float(self.weights[locked].sum())
        free = max(0.0, 1 - fixed)
        low, high = 0.0, free
        for _ in range(45):
            fee = (low + high) / 2
            turnover = np.abs((free - fee) * action[:-1] - self.weights[indices]).sum()
            if fee > self.cost_rate * turnover:
                high = fee
            else:
                low = fee
        fee = (low + high) / 2
        targets = np.zeros_like(self.weights)
        targets[locked] = self.weights[locked]
        targets[indices] = (free - fee) * action[:-1]
        cash = (free - fee) * action[-1]
        used = targets > 0
        returns = self.market.returns[t]
        if not np.isfinite(returns[used]).all():
            raise ValueError(f"Unspecified research return at {self.market.dates[t]}")
        values = np.zeros_like(targets)
        values[used] = targets[used] * (1 + returns[used])
        cash *= 1 + self.market.cash_returns[t]
        settle = self.market.cash_settlements[t]
        cash += float(values[settle].sum())
        values[settle] = 0
        growth = float(values.sum() + cash)
        if not np.isfinite(growth) or growth < 0:
            raise ValueError("Invalid research equity")
        turnover = float(np.abs(targets[indices] - self.weights[indices]).sum())
        assumed_exposure = float(targets[self.market.assumed_exits[t]].sum())
        old_equity = self.equity
        self.equity *= growth
        self.weights = values / growth if growth > 0 else np.zeros_like(values)
        self.cash = cash / growth if growth > 0 else 0.0
        self.position += 1
        done = growth <= 0 or self.position >= self.right
        return float(np.log(max(growth, 1e-12))), done, {
            "date": str(self.market.dates[t]), "return": growth - 1,
            "benchmark_return": float(self.market.benchmark[t]),
            "turnover": turnover, "cost_fraction": fee, "equity": self.equity,
            "cash_weight": self.cash, "bankrupt": growth <= 0,
            "locked_weight": fixed, "assumed_exit_weight": assumed_exposure,
            "assumed_exit_capital": old_equity * assumed_exposure,
        }
