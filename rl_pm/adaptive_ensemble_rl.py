from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from rl_pm.execution_train import update_network
from rl_pm.index_exposure_rl import PRICE_FILE, dated_bounds, load_series


ROOT = Path(__file__).resolve().parents[1]
DECISION_INTERVAL = 5
WEIGHT_PROFILES = np.asarray([
    [0.20, 0.40, 0.20, 0.20],  # Original high-volatility profile
    [0.40, 0.20, 0.20, 0.20],  # Original strong-trend profile
    [0.25, 0.25, 0.25, 0.25],  # Original balanced profile
], dtype=np.float32)
PROFILE_NAMES = ("mean_reversion", "momentum", "balanced")


class WeightQNetwork(nn.Module):
    def __init__(self, state_size: int, action_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, states):
        return self.layers(states)


class ContinuousWeightPolicy(nn.Module):
    def __init__(self, state_size: int, component_count: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, component_count)
        self.critic = nn.Linear(64, 1)

    def distribution_and_value(self, state):
        encoded = self.encoder(state)
        concentration = torch.nn.functional.softplus(self.actor(encoded)) + 0.5
        return torch.distributions.Dirichlet(concentration), self.critic(encoded).squeeze(-1)

    def act(self, state, deterministic=False):
        distribution, value = self.distribution_and_value(state)
        weights = distribution.mean if deterministic else distribution.sample()
        return weights, distribution.log_prob(weights), value


def train_continuous_agent(components, market_state, finite, returns, bounds, episodes, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    left, right = bounds
    decisions = [
        index for index in range(left, right - DECISION_INTERVAL, DECISION_INTERVAL)
        if finite[index] and finite[index + DECISION_INTERVAL]
    ]
    model = ContinuousWeightPolicy(market_state.shape[1] + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    history = []

    for episode in range(episodes):
        length = rng.randint(20, min(50, len(decisions) - 1))
        start = rng.randint(0, len(decisions) - length - 1)
        records = []
        prior_exposure = 0.0
        episode_reward = 0.0
        for step in range(length):
            index = decisions[start + step]
            next_index = index + DECISION_INTERVAL
            state = state_for(index, market_state, prior_exposure)
            state_tensor = torch.from_numpy(state)
            with torch.no_grad():
                weights, log_probability, value = model.act(state_tensor)
            weight_array = weights.numpy()
            score = float(components[index] @ weight_array)
            exposure = target_exposure(score)
            period_return = float(np.prod(1.0 + exposure * returns[index + 1 : next_index + 1]) - 1.0)
            turnover_cost = 0.0005 * abs(exposure - prior_exposure)
            reward = float(np.log(max(1e-8, (1.0 + period_return) * (1.0 - turnover_cost))) * 100.0)
            done = step + 1 == length
            records.append({
                "state": state_tensor,
                "weights": weights,
                "log_probability": float(log_probability),
                "value": float(value),
                "reward": reward,
                "done": done,
            })
            prior_exposure = exposure
            episode_reward += reward

        advantage = 0.0
        next_value = 0.0
        for record in reversed(records):
            continuation = 0.0 if record["done"] else 1.0
            delta = record["reward"] + 0.97 * next_value * continuation - record["value"]
            advantage = delta + 0.97 * 0.9 * continuation * advantage
            record["advantage"] = advantage
            record["target"] = advantage + record["value"]
            next_value = record["value"]
        advantages = np.asarray([record["advantage"] for record in records])
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        for record, normalized in zip(records, advantages):
            record["advantage"] = float(normalized)

        for _ in range(3):
            order = list(range(len(records)))
            rng.shuffle(order)
            for offset in range(0, len(order), 32):
                batch = [records[i] for i in order[offset : offset + 32]]
                probabilities = []
                values = []
                entropies = []
                for record in batch:
                    distribution, value = model.distribution_and_value(record["state"])
                    probabilities.append(distribution.log_prob(record["weights"]))
                    values.append(value.squeeze())
                    entropies.append(distribution.entropy())
                new_log = torch.stack(probabilities)
                old_log = torch.tensor([r["log_probability"] for r in batch], dtype=torch.float32)
                advantage_batch = torch.tensor([r["advantage"] for r in batch], dtype=torch.float32)
                targets = torch.tensor([r["target"] for r in batch], dtype=torch.float32)
                ratio = torch.exp((new_log - old_log).clamp(-20, 20))
                clipped = ratio.clamp(0.8, 1.2)
                policy_loss = -torch.minimum(ratio * advantage_batch, clipped * advantage_batch).mean()
                value_loss = torch.nn.functional.mse_loss(torch.stack(values), targets)
                entropy_bonus = torch.stack(entropies).mean()
                loss = policy_loss + 0.5 * value_loss - 0.005 * entropy_bonus
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        history.append({"episode": episode + 1, "reward": episode_reward})

    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "adaptive_weights_ppo.pt")
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward"])
        writer.writeheader()
        writer.writerows(history)
    return model


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(window - 1, len(values)):
        result[index] = np.std(values[index - window + 1 : index + 1], ddof=1)
    return result


def _rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        less = np.sum(sample < values[index])
        tied = np.sum(sample == values[index])
        result[index] = (less + (tied + 1) / 2) / window
    return result


def copied_stockalgo_signals(close: np.ndarray, returns: np.ndarray):
    """Local research copy of the signal math and ensemble weights in StockAlgo."""
    close = np.asarray(close, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    n = len(close)

    def mean(values, window):
        return np.convolve(values, np.ones(window) / window, mode="full")[:n]

    mom_21 = close / np.roll(close, 20) - 1.0
    mom_63 = close / np.roll(close, 62) - 1.0
    mom_126 = close / np.roll(close, 125) - 1.0
    mom_21[:20] = np.nan
    mom_63[:62] = np.nan
    mom_126[:125] = np.nan

    vol_20 = _rolling_std(returns, 20) * np.sqrt(252)
    vol_126 = _rolling_std(returns, 126) * np.sqrt(252)
    vol_rank = _rolling_percentile(vol_20, 500)
    log_returns = np.zeros(n)
    log_returns[1:] = np.log1p(returns[1:])
    mean_reversion = -(log_returns - mean(log_returns, 20))

    delta = np.diff(close, prepend=close[0])
    gains = mean(np.maximum(delta, 0.0), 14)
    losses = mean(np.maximum(-delta, 0.0), 14)
    rs = np.divide(gains, losses, out=np.full(n, np.inf), where=losses > 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_reversion = (50 - rsi) / 50

    sma_20 = mean(close, 20)
    sma_50 = mean(close, 50)
    sma_200 = mean(close, 200)
    std_20 = _rolling_std(close, 20)
    band_width = 4 * std_20
    bb_position = np.divide(
        close - (sma_20 - 2 * std_20),
        band_width,
        out=np.full(n, 0.5),
        where=band_width > 0,
    )
    value_score = np.mean(
        np.column_stack([
            close / sma_50 - 1.0,
            close / sma_200 - 1.0,
            1.0 - bb_position,
        ]),
        axis=1,
    )
    trend_strength = np.abs(sma_20 - sma_50) / np.maximum(sma_50, 1e-12)
    price_momentum = np.divide(
        mom_126,
        np.where(np.isfinite(vol_126) & (vol_126 > 0), vol_126, 0.2),
    )
    earnings_proxy = mom_21 - mom_63
    momentum_score = 0.6 * price_momentum + 0.4 * earnings_proxy
    reversion_score = 0.6 * mean_reversion + 0.4 * rsi_reversion

    components = np.column_stack([
        momentum_score,
        reversion_score,
        value_score,
        np.zeros(n),  # StockAlgo currently reads "knn_prediction", but KNN returns "ml_signal".
    ])
    state = np.column_stack([
        np.clip(returns / 0.03, -5, 5),
        np.clip((close / np.roll(close, 5) - 1.0) / 0.06, -5, 5),
        np.clip(mom_21 / 0.12, -5, 5),
        np.clip(mom_63 / 0.20, -5, 5),
        np.clip(vol_20 / 0.05, 0, 5),
        np.clip(vol_126 / 0.05, 0, 5),
        np.clip(close / np.maximum.accumulate(close) - 1.0, -5, 0),
        np.clip(close / sma_50 - 1.0, -5, 5),
        np.clip(close / sma_200 - 1.0, -5, 5),
        np.clip(vol_rank, 0, 1),
        np.clip(trend_strength / 0.05, 0, 5),
        ((rsi > 70) | (rsi < 30)).astype(np.float64),
        np.clip(price_momentum, -5, 5),
        np.clip(earnings_proxy / 0.1, -5, 5),
        np.clip(mean_reversion / 0.02, -5, 5),
        np.clip(rsi_reversion, -5, 5),
        np.clip(value_score / 0.2, -5, 5),
    ]).astype(np.float32)
    state[0, :4] = np.nan
    state[:125, 2] = np.nan
    state[:62, 3] = np.nan
    state[:499, 9] = np.nan
    state[:200, 8] = np.nan
    finite = np.isfinite(state).all(axis=1) & np.isfinite(components).all(axis=1)

    # Reproduce the original rule-based selector and its three weight vectors.
    original_profile = np.full(n, 2, dtype=np.int64)
    original_profile[vol_rank > 0.7] = 0
    original_profile[(vol_rank <= 0.7) & (trend_strength > 0.05)] = 1
    original_score = np.sum(components * WEIGHT_PROFILES[original_profile], axis=1)
    return components.astype(np.float32), state, finite, original_profile, original_score.astype(np.float32)


def target_exposure(score):
    # Preserve StockAlgo's BUY/HOLD/SELL threshold in a long-only SPY test.
    return 1.0 if score > 0.02 else 0.0


def state_for(index, market_state, prior_exposure):
    return np.concatenate([market_state[index], [prior_exposure]]).astype(np.float32)


def score_for_action(index, action, components):
    return float(components[index] @ WEIGHT_PROFILES[action])


def train_agent(components, market_state, finite, returns, bounds, episodes, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    left, right = bounds
    decisions = list(range(left, right - DECISION_INTERVAL, DECISION_INTERVAL))
    decisions = [index for index in decisions if finite[index] and finite[index + DECISION_INTERVAL]]
    network = WeightQNetwork(market_state.shape[1] + 1, len(WEIGHT_PROFILES))
    target = WeightQNetwork(market_state.shape[1] + 1, len(WEIGHT_PROFILES))
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)
    replay = deque(maxlen=30_000)
    epsilon = 1.0
    total_steps = 0
    history = []

    for episode in range(episodes):
        length = rng.randint(20, min(50, len(decisions) - 1))
        start = rng.randint(0, len(decisions) - length - 1)
        prior_exposure = 0.0
        episode_reward = 0.0
        for step in range(length):
            index = decisions[start + step]
            next_index = index + DECISION_INTERVAL
            state = state_for(index, market_state, prior_exposure)
            if rng.random() < epsilon:
                action = rng.randrange(len(WEIGHT_PROFILES))
            else:
                with torch.no_grad():
                    action = int(network(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
            exposure = target_exposure(score_for_action(index, action, components))
            period_return = float(np.prod(1.0 + exposure * returns[index + 1 : next_index + 1]) - 1.0)
            turnover_cost = 0.0005 * abs(exposure - prior_exposure)
            reward = float(np.log(max(1e-8, (1.0 + period_return) * (1.0 - turnover_cost))) * 100.0)
            next_state = state_for(next_index, market_state, exposure)
            done = step + 1 == length
            replay.append((state, action, reward, next_state, done))
            prior_exposure = exposure
            episode_reward += reward
            total_steps += 1
            epsilon = max(0.05, epsilon * 0.9997)
            if len(replay) >= 128:
                update_network(network, target, optimizer, rng.sample(replay, 128), 0.97)
            if total_steps % 500 == 0:
                target.load_state_dict(network.state_dict())
        history.append({"episode": episode + 1, "reward": episode_reward, "epsilon": epsilon})
        if (episode + 1) % 25 == 0:
            target.load_state_dict(network.state_dict())

    output.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output / "adaptive_weights_dqn.pt")
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(history)
    return network


def evaluate(dates, returns, components, market_state, finite, bounds, network, original_profile, split, agent_type="dqn"):
    left, right = bounds
    records = []
    for policy in ("rl_weights", "stockalgo_original", "balanced_weights"):
        wealth = 1.0
        peak = 1.0
        prior_exposure = 0.0
        for index in range(left, right - DECISION_INTERVAL, DECISION_INTERVAL):
            if not finite[index] or not finite[index + DECISION_INTERVAL]:
                continue
            if policy == "rl_weights":
                state = state_for(index, market_state, prior_exposure)
                with torch.no_grad():
                    if agent_type == "dqn":
                        profile = int(network(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
                        weights = WEIGHT_PROFILES[profile]
                        profile_name = PROFILE_NAMES[profile]
                    else:
                        weights, _, _ = network.act(torch.from_numpy(state), deterministic=True)
                        weights = weights.numpy()
                        profile_name = "continuous_ppo"
            elif policy == "stockalgo_original":
                profile = int(original_profile[index])
                weights = WEIGHT_PROFILES[profile]
                profile_name = PROFILE_NAMES[profile]
            else:
                profile = 2
                weights = WEIGHT_PROFILES[profile]
                profile_name = PROFILE_NAMES[profile]
            score = float(components[index] @ weights)
            exposure = target_exposure(score)
            next_index = index + DECISION_INTERVAL
            market_return = float(np.prod(1.0 + exposure * returns[index + 1 : next_index + 1]) - 1.0)
            turnover = abs(exposure - prior_exposure)
            cost = 0.0005 * turnover
            growth = (1.0 + market_return) * (1.0 - cost)
            wealth *= growth
            peak = max(peak, wealth)
            records.append({
                "split": split,
                "date": str(dates[index]),
                "policy": policy,
                "weight_profile": profile_name,
                "momentum_weight": float(weights[0]),
                "mean_reversion_weight": float(weights[1]),
                "value_weight": float(weights[2]),
                "ml_weight": float(weights[3]),
                "ensemble_score": score,
                "exposure": exposure,
                "return": growth - 1.0,
                "turnover": turnover,
                "wealth": wealth,
                "drawdown": wealth / peak - 1.0,
            })
            prior_exposure = exposure
    return records


def summarize(records):
    result = {}
    for policy in sorted({row["policy"] for row in records}):
        rows = [row for row in records if row["policy"] == policy]
        values = np.asarray([row["return"] for row in rows], dtype=np.float64)
        wealth = np.asarray([row["wealth"] for row in rows], dtype=np.float64)
        std = float(np.std(values, ddof=1))
        annualizer = np.sqrt(252 / DECISION_INTERVAL)
        years = len(rows) * DECISION_INTERVAL / 252
        result[policy] = {
            "total_return_pct": float((wealth[-1] - 1) * 100),
            "cagr_pct": float((wealth[-1] ** (1 / years) - 1) * 100),
            "annualized_volatility_pct": std * annualizer * 100,
            "sharpe_zero_rate": float(np.mean(values) / std * annualizer) if std > 0 else 0.0,
            "max_drawdown_pct": float(min(row["drawdown"] for row in rows) * 100),
            "mean_exposure_pct": float(np.mean([row["exposure"] for row in rows]) * 100),
            "annual_turnover": float(sum(row["turnover"] for row in rows) / years),
            "profile_frequency_pct": {
                name: float(np.mean([row["weight_profile"] == name for row in rows]) * 100)
                for name in PROFILE_NAMES
            },
            "average_weights": {
                name: float(np.mean([row[column] for row in rows]))
                for name, column in (
                    ("momentum", "momentum_weight"),
                    ("mean_reversion", "mean_reversion_weight"),
                    ("value", "value_weight"),
                    ("ml", "ml_weight"),
                )
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="DQN learns to select StockAlgo ensemble weight profiles.")
    parser.add_argument("--prices", type=Path, default=PRICE_FILE)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--agent", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--output", type=Path, default=Path("outputs/adaptive-ensemble-dqn-001"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    dates, close, returns, market_features = load_series(args.prices)
    components, copied_state, finite, original_profile, original_score = copied_stockalgo_signals(close, returns)
    state = np.column_stack([market_features, copied_state[:, 9:]]).astype(np.float32)
    finite &= np.isfinite(state).all(axis=1)
    train_bounds = dated_bounds(dates, "1996-01-01", "2012-12-31")
    validation_bounds = dated_bounds(dates, "2013-01-01", "2018-12-31")
    test_bounds = dated_bounds(dates, "2019-01-01", str(dates[-1]))
    first_valid = int(np.flatnonzero(finite)[0])
    train_bounds = (max(train_bounds[0], first_valid), train_bounds[1])
    if train_bounds[1] > validation_bounds[0] or validation_bounds[1] > test_bounds[0]:
        raise ValueError("Chronological splits overlap")

    trainer = train_agent if args.agent == "dqn" else train_continuous_agent
    network = trainer(components, state, finite, returns, train_bounds, args.episodes, args.seed, args.output)
    records = []
    records.extend(evaluate(dates, returns, components, state, finite, validation_bounds, network, original_profile, "validation", args.agent))
    records.extend(evaluate(dates, returns, components, state, finite, test_bounds, network, original_profile, "test", args.agent))
    with (args.output / "weekly_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = {
        split: summarize([row for row in records if row["split"] == split])
        for split in ("validation", "test")
    }
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "run.json").write_text(json.dumps({
        "source_copy": "StockAlgo ensemble.py signal math and ensemble weight profiles, reproduced locally; original file is unchanged",
        "algorithm": (
            "Double DQN selects one of the original three rule-based weight profiles every five sessions"
            if args.agent == "dqn"
            else "PPO outputs continuous Dirichlet-distributed weights over all four copied components every five sessions"
        ),
        "agent": args.agent,
        "component_order": ["momentum", "mean_reversion", "value", "ml"],
        "weight_profiles": {name: weights.tolist() for name, weights in zip(PROFILE_NAMES, WEIGHT_PROFILES)},
        "source_ml_behavior": "kept unchanged: the source aggregator reads knn_prediction while its KNN function returns ml_signal, so the weighted ML component is zero",
        "trade_logic": "unchanged source threshold: buy above ensemble score 0.02; long-only evaluation holds cash otherwise",
        "observation_timing": "signal uses data through close t; reward is adjusted SPY return over following five sessions",
        "training_period": [str(dates[train_bounds[0]]), str(dates[train_bounds[1] - 1])],
        "validation_period": [str(dates[validation_bounds[0]]), str(dates[validation_bounds[1] - 1])],
        "test_period": [str(dates[test_bounds[0]]), str(dates[test_bounds[1] - 1])],
        "episodes": args.episodes,
        "seed": args.seed,
        "cost_bps_per_unit_turnover": 5,
        "price_source": str(args.prices.resolve().relative_to(ROOT)) if args.prices.resolve().is_relative_to(ROOT) else args.prices.name,
        "limitations": ["single SPY series rather than the full stock universe", "one training seed", "cash earns zero", "historical outputs are research estimates"],
    }, indent=2) + "\n")

    test_rows = [row for row in records if row["split"] == "test"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for policy in ("rl_weights", "stockalgo_original", "balanced_weights"):
        rows = [row for row in test_rows if row["policy"] == policy]
        axes[0].plot([np.datetime64(row["date"]) for row in rows], [row["wealth"] for row in rows], label=policy.replace("_", " "))
    axes[0].set_title("Adaptive ensemble weights vs. unchanged source rule")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    rows = [row for row in test_rows if row["policy"] == "rl_weights"]
    for column, name in (("momentum_weight", "Momentum"), ("mean_reversion_weight", "Mean reversion"), ("value_weight", "Value")):
        axes[1].step([np.datetime64(row["date"]) for row in rows], [float(row[column]) for row in rows], where="post", label=name)
    axes[1].set_title("Weights selected by the RL agent")
    axes[1].set_ylabel("Weight")
    axes[1].set_ylim(0, 0.55)
    axes[1].legend(ncol=3)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "adaptive_weights_test.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))
    print(f"Results: {args.output / 'weekly_results.csv'}")


if __name__ == "__main__":
    main()
