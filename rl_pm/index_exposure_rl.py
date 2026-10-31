from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from rl_pm.execution_train import QNetwork, update_network


ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / "data/raw/final/prices.parquet"
ACTIONS = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
FEATURE_NAMES = (
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    "volatility_21d",
    "volatility_63d",
    "drawdown_126d",
    "sma_50_distance",
    "sma_200_distance",
)


def load_series(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing price file: {path}")
    connection = duckdb.connect()
    rows = connection.execute(
        """
        SELECT date, adjusted_close
        FROM read_parquet(?)
        WHERE security_id='SPY' AND adjusted_close > 0
        ORDER BY date
        """,
        [str(path)],
    ).fetchall()
    connection.close()
    dates = np.asarray([row[0] for row in rows], dtype="datetime64[D]")
    close = np.asarray([row[1] for row in rows], dtype=np.float64)
    returns = np.full(len(close), np.nan, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0

    def rolling_mean(values, window):
        out = np.full(len(values), np.nan)
        for index in range(window - 1, len(values)):
            out[index] = np.mean(values[index - window + 1 : index + 1])
        return out

    def rolling_std(values, window):
        out = np.full(len(values), np.nan)
        for index in range(window, len(values)):
            out[index] = np.std(values[index - window + 1 : index + 1], ddof=1)
        return out

    features = np.column_stack(
        [
            returns,
            close / np.roll(close, 5) - 1.0,
            close / np.roll(close, 21) - 1.0,
            close / np.roll(close, 63) - 1.0,
            rolling_std(returns, 21),
            rolling_std(returns, 63),
            np.full(len(close), np.nan),
            close / rolling_mean(close, 50) - 1.0,
            close / rolling_mean(close, 200) - 1.0,
        ]
    )
    for window, column in ((5, 1), (21, 2), (63, 3)):
        features[:window, column] = np.nan
    rolling_high = np.full(len(close), np.nan)
    for index in range(125, len(close)):
        rolling_high[index] = np.max(close[index - 125 : index + 1])
    features[:, 6] = close / rolling_high - 1.0
    features[0, 0] = np.nan
    feature_scales = np.asarray([0.03, 0.06, 0.12, 0.20, 0.05, 0.05, 0.30, 0.15, 0.25])
    features = np.clip(features / feature_scales, -5.0, 5.0).astype(np.float32)
    return dates, close, returns, features


def dated_bounds(dates, start, end):
    left = int(np.searchsorted(dates, np.datetime64(start), side="left"))
    right = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    if right - left < 3:
        raise ValueError(f"Insufficient prices in {start} through {end}")
    return left, right


def make_state(features, exposure):
    return np.concatenate((features, np.asarray([exposure], dtype=np.float32)))


def train(features, returns, train_bounds, episodes, seed, output, interval: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    left, right = train_bounds
    network = QNetwork(features.shape[1] + 1)
    target = QNetwork(features.shape[1] + 1)
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)
    replay = deque(maxlen=40_000)
    epsilon = 1.0
    history = []
    total_steps = 0
    if interval < 1:
        raise ValueError("decision interval must be positive")
    max_steps = min(252 // interval, (right - left - 1) // interval)
    min_steps = min(max(1, int(np.ceil(63 / interval))), max_steps)

    for episode in range(episodes):
        length_steps = rng.randint(min_steps, max_steps)
        length = length_steps * interval
        start = rng.randint(left, right - length - 1)
        exposure = 0.0
        episode_reward = 0.0
        for step in range(length_steps):
            index = start + step * interval
            next_index = index + interval
            state = make_state(features[index], exposure)
            if rng.random() < epsilon:
                action_index = rng.randrange(len(ACTIONS))
            else:
                with torch.no_grad():
                    q_values = network(torch.tensor(state).unsqueeze(0))
                action_index = int(q_values.argmax(dim=1).item())
            target_exposure = float(ACTIONS[action_index])
            period_return = np.prod(
                1.0 + target_exposure * returns[index + 1 : next_index + 1]
            ) - 1.0
            net_growth = (1.0 + period_return) * max(
                1e-8, 1.0 - 0.0005 * abs(target_exposure - exposure)
            )
            reward = float(np.log(max(1e-8, net_growth))) * 100.0
            next_state = make_state(features[next_index], target_exposure)
            done = step + 1 >= length_steps
            replay.append((state, action_index, reward, next_state, done))
            episode_reward += reward
            exposure = target_exposure
            total_steps += 1
            epsilon = max(0.04, epsilon * 0.99985)

            if len(replay) >= 128:
                batch = rng.sample(replay, 128)
                update_network(network, target, optimizer, batch, 0.99)
            if total_steps % 500 == 0:
                target.load_state_dict(network.state_dict())

        history.append({"episode": episode + 1, "reward": episode_reward, "epsilon": epsilon})
        if (episode + 1) % 25 == 0:
            target.load_state_dict(network.state_dict())

    output.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output / "exposure_dqn.pt")
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(history)
    return network, history


def vol_target_exposure(features, index, target_vol=0.10):
    daily_vol = float(features[index, 4]) * 0.05
    if daily_vol <= 0 or not np.isfinite(daily_vol):
        return 0.0
    annualized = daily_vol * np.sqrt(252)
    return float(np.clip(target_vol / annualized, 0.0, 1.0))


def evaluate(dates, returns, features, bounds, network, label):
    left, right = bounds
    policies = ("dqn", "buy_hold", "half_exposure", "cash", "volatility_target", "trend_200d")
    records = []
    for policy in policies:
        wealth = 1.0
        exposure = 0.0
        peak = 1.0
        for index in range(left, right - 1):
            if policy == "dqn":
                state = torch.tensor(make_state(features[index], exposure)).unsqueeze(0)
                with torch.no_grad():
                    action_index = int(network(state).argmax(dim=1).item())
                next_exposure = float(ACTIONS[action_index])
            elif policy == "buy_hold":
                next_exposure = 1.0
            elif policy == "half_exposure":
                next_exposure = 0.5
            elif policy == "cash":
                next_exposure = 0.0
            elif policy == "volatility_target":
                next_exposure = vol_target_exposure(features, index)
            else:
                next_exposure = 1.0 if features[index, 8] > 0 else 0.0

            transaction_cost = 0.0005 * abs(next_exposure - exposure)
            net_return = next_exposure * returns[index + 1] - transaction_cost
            wealth *= max(1e-8, 1.0 + net_return)
            peak = max(peak, wealth)
            records.append(
                {
                    "split": label,
                    "date": str(dates[index + 1]),
                    "policy": policy,
                    "exposure": next_exposure,
                    "daily_return": net_return,
                    "turnover": abs(next_exposure - exposure),
                    "wealth": wealth,
                    "drawdown": wealth / peak - 1.0,
                }
            )
            exposure = next_exposure
    return records


def summarize(records):
    report = {}
    for policy in sorted({row["policy"] for row in records}):
        rows = [row for row in records if row["policy"] == policy]
        rets = np.asarray([row["daily_return"] for row in rows], dtype=np.float64)
        wealth = np.asarray([row["wealth"] for row in rows], dtype=np.float64)
        years = len(rets) / 252.0
        cagr = float(wealth[-1] ** (1 / years) - 1) if years > 0 and wealth[-1] > 0 else -1.0
        volatility = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)) if np.std(rets, ddof=1) > 0 else 0.0
        report[policy] = {
            "sessions": len(rets),
            "total_return_pct": float((wealth[-1] - 1.0) * 100),
            "cagr_pct": cagr * 100,
            "annualized_volatility_pct": volatility * 100,
            "sharpe_zero_rate": sharpe,
            "max_drawdown_pct": float(min(row["drawdown"] for row in rows) * 100),
            "mean_exposure_pct": float(np.mean([row["exposure"] for row in rows]) * 100),
            "annual_turnover": float(sum(row["turnover"] for row in rows) / years) if years else 0.0,
        }
    return report


def save_outputs(output, dates, returns, features, splits, network, history, metadata):
    records = []
    for label, bounds in splits.items():
        records.extend(evaluate(dates, returns, features, bounds, network, label))
    with (output / "daily_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = {
        label: summarize([row for row in records if row["split"] == label])
        for label in splits
    }
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    test_rows = [row for row in records if row["split"] == "test"]
    for policy in sorted({row["policy"] for row in test_rows}):
        rows = [row for row in test_rows if row["policy"] == policy]
        axes[0].plot(
            [row["date"] for row in rows],
            [row["wealth"] for row in rows],
            label=policy,
            linewidth=1.2,
        )
    axes[0].set_title("Untouched test period")
    axes[0].set_ylabel("Growth of $1")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    dqn = [row for row in test_rows if row["policy"] == "dqn"]
    axes[1].plot([row["date"] for row in dqn], [row["exposure"] for row in dqn], color="#5B8E7D")
    axes[1].set_title("DQN SPY exposure")
    axes[1].set_ylabel("Target exposure")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "exposure_test.png", dpi=170, facecolor="white")
    plt.close(fig)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Train an RL exposure overlay on historical SPY prices.")
    parser.add_argument("--prices", type=Path, default=PRICE_FILE)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, default=Path("outputs/index-exposure-dqn-001"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    dates, close, returns, features = load_series(args.prices)
    if not np.isfinite(features).all() or not np.isfinite(returns[1:]).all():
        valid = np.isfinite(features).all(axis=1) & np.isfinite(returns)
        first_valid = int(np.flatnonzero(valid)[0])
    else:
        first_valid = 1
    train_bounds = dated_bounds(dates, "1996-01-01", "2012-12-31")
    validation_bounds = dated_bounds(dates, "2013-01-01", "2018-12-31")
    test_bounds = dated_bounds(dates, "2019-01-01", str(dates[-1]))
    train_bounds = (max(train_bounds[0], first_valid), train_bounds[1])
    if train_bounds[1] > validation_bounds[0] or validation_bounds[1] > test_bounds[0]:
        raise ValueError("Train, validation, and test splits overlap")
    network, history = train(
        features,
        returns,
        train_bounds,
        args.episodes,
        args.seed,
        args.output,
    )
    with (args.output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "algorithm": "Double DQN; discrete target exposure [0, 0.5, 1]",
        "instrument": "SPY adjusted close used as an S&P 500 total-return proxy",
        "features": list(FEATURE_NAMES),
        "decision_timing": "observe through close t; target exposure earns adjusted return t to t+1",
        "transaction_cost_bps_per_unit_turnover": 5,
        "training_period": [str(dates[train_bounds[0]]), str(dates[train_bounds[1] - 1])],
        "validation_period": [str(dates[validation_bounds[0]]), str(dates[validation_bounds[1] - 1])],
        "test_period": [str(dates[test_bounds[0]]), str(dates[test_bounds[1] - 1])],
        "training_episodes": args.episodes,
        "seed": args.seed,
        "future_information_used_in_features": False,
        "source": str(args.prices),
        "limitations": [
            "single broad-market ETF exposure task, not constituent selection",
            "cash earns zero",
            "SPY inception limits this experiment to 1995 onward",
            "synthetic terminal-return recovery rules are not used",
            "one chronological holdout; no claim of deployable edge",
        ],
    }
    summaries = save_outputs(
        args.output,
        dates,
        returns,
        features,
        {"validation": validation_bounds, "test": test_bounds},
        network,
        history,
        metadata,
    )
    print(json.dumps(summaries, indent=2))
    print(f"Results: {args.output / 'daily_results.csv'}")
    print(f"Chart: {args.output / 'exposure_test.png'}")


if __name__ == "__main__":
    main()
