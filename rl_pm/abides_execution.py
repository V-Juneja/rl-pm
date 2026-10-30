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


def clean_state(observation: np.ndarray) -> np.ndarray:
    state = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    state[5] *= 100.0
    state[6] /= 10.0
    state[7:] /= 100.0
    return np.clip(state, -10.0, 10.0)
