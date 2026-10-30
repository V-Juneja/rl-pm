from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np


def load_environment_source():
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "data" / "external" / "abides-ng" / "abides-gym"
    if not source.is_dir():
        raise RuntimeError(
            "ABIDES-NG is missing. Run scripts/setup_execution_env.sh in the project environment."
        )
    sys.path.insert(0, str(source))

    if importlib.util.find_spec("ray") is None:
        ray = types.ModuleType("ray")
        tune = types.ModuleType("ray.tune")
        registry = types.ModuleType("ray.tune.registry")
        registry.register_env = lambda *_args, **_kwargs: None
        ray.tune = tune
        tune.registry = registry
        sys.modules.update(
            {"ray": ray, "ray.tune": tune, "ray.tune.registry": registry}
        )

    from abides_gym.envs.markets_execution_environment_v0 import (
        SubGymMarketsExecutionEnv_v0,
    )

    class ExecutionEnv(SubGymMarketsExecutionEnv_v0):
        def reset(self, *, seed: int | None = None, options=None):
            self.pnl = 0
            self.step_index = 0
            self.custom_metrics_tracker = self.CustomMetricsTracker()
            self.custom_metrics_tracker.action_counter = {
                f"action_{action}": 0 for action in range(3)
            }
            return super().reset(seed=seed, options=options)

    return ExecutionEnv


def load_daily_investor_source():
    load_environment_source()
    from abides_gym.envs.markets_daily_investor_environment_v0 import (
        SubGymMarketsDailyInvestorEnv_v0,
    )
    import gymnasium as gym
    import abides_markets.agents.utils as markets_agent_utils
    from abides_core.utils import str_to_ns

    class IntradayEnv(SubGymMarketsDailyInvestorEnv_v0):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.state_lows = np.concatenate(
                [self.state_lows.reshape(-1), np.array([0.0], dtype=np.float32)]
            ).reshape(-1, 1)
            self.state_highs = np.concatenate(
                [self.state_highs.reshape(-1), np.array([1.0], dtype=np.float32)]
            ).reshape(-1, 1)
            self.num_state_features += 1
            self.observation_space = gym.spaces.Box(
                self.state_lows,
                self.state_highs,
                shape=(self.num_state_features, 1),
                dtype=np.float32,
            )

        def reset(self, *, seed: int | None = None, options=None):
            self.previous_marked_to_market = self.starting_cash
            return super().reset(seed=seed, options=options)

        @markets_agent_utils.ignore_mkt_data_buffer_decorator
        def raw_state_to_state(self, raw_state):
            from abides_markets.agents.utils import get_imbalance, get_mid_price

            bids = raw_state["parsed_mkt_data"]["bids"]
            asks = raw_state["parsed_mkt_data"]["asks"]
            transactions = raw_state["parsed_mkt_data"]["last_transaction"]
            holdings = raw_state["internal_data"]["holdings"][-1]
            imbalance = get_imbalance(bids[-1], asks[-1], depth=3)
            mid_prices = [
                get_mid_price(bid, ask, transaction)
                for bid, ask, transaction in zip(bids, asks, transactions)
            ]
            spread = (
                (asks[-1][0][0] if asks[-1] else mid_prices[-1])
                - (bids[-1][0][0] if bids[-1] else mid_prices[-1])
            )
            direction = mid_prices[-1] - transactions[-1]
            returns = np.diff(mid_prices)
            padded_returns = np.zeros(self.state_history_length - 1)
            if len(returns):
                padded_returns[-min(len(returns), len(padded_returns)) :] = returns[
                    -len(padded_returns) :
                ]
            base = np.asarray(
                [holdings, imbalance, spread, direction, *padded_returns],
                dtype=np.float32,
            )
            internal = raw_state["internal_data"]
            elapsed = (
                internal["current_time"][-1]
                - internal["mkt_open"][-1]
                - self.first_interval
            )
            duration = (
                self.mkt_close - str_to_ns("09:30:00") - self.first_interval
            )
            time_pct = float(np.clip(elapsed / duration, 0.0, 1.0))
            return np.append(base, time_pct).astype(np.float32).reshape(-1, 1)

    return IntradayEnv


def clean_state(observation: np.ndarray) -> np.ndarray:
    state = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    state[5] *= 100.0
    state[6] /= 10.0
    state[7:] /= 100.0
    return np.clip(state, -10.0, 10.0)
