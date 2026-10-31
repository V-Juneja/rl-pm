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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from rl_pm.adaptive_ensemble_rl import (
    PROFILE_NAMES,
    WEIGHT_PROFILES,
    copied_stockalgo_signals,
)
from rl_pm.execution_train import update_network
from rl_pm.index_exposure_rl import dated_bounds


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "outputs/training-staging-v3"
RESEARCH = ROOT / "data/processed/research-stress"
PRICES = ROOT / "data/raw/final-extended/prices.parquet"
MEMBERSHIP = ROOT / "data/raw/final-extended/membership.parquet"
INTERVAL = 5


class WeightQNetwork(nn.Module):
    def __init__(self, state_size, action_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, states):
        return self.layers(states)


def load_close_panel(price_path, membership_path, dates, ids, tradable):
    panel = np.full(tradable.shape, np.nan, dtype=np.float32)
    connection = duckdb.connect()
    query = """
        SELECT p.date, p.security_id, p.adjusted_close
        FROM read_parquet(?) AS p
        WHERE p.date >= ? AND p.date <= ?
          AND p.security_id IN (
              SELECT DISTINCT security_id FROM read_parquet(?)
          )
    """
    result = connection.execute(
        query,
        [str(price_path), str(dates[0]), str(dates[-1]), str(membership_path)],
    )
    date_lookup = {str(day): index for index, day in enumerate(dates)}
    id_lookup = {str(sid): index for index, sid in enumerate(ids)}
    loaded = 0
    while True:
        batch = result.fetchmany(250_000)
        if not batch:
            break
        rows = []
        cols = []
        vals = []
        for day, sid, value in batch:
            row = date_lookup.get(str(day))
            col = id_lookup.get(str(sid))
            if row is not None and col is not None and np.isfinite(value) and value > 0:
                rows.append(row)
                cols.append(col)
                vals.append(value)
        if rows:
            panel[np.asarray(rows), np.asarray(cols)] = np.asarray(vals, dtype=np.float32)
            loaded += len(rows)
        if loaded and loaded % 1_000_000 < 250_000:
            print(f"Loaded {loaded:,} dated prices", flush=True)
    connection.close()
    panel[~tradable] = np.nan
    print(f"Price panel ready: {loaded:,} observations", flush=True)
    return panel


def build_signal_panel(close, dates, ids, decisions, tradable):
    signal_dates = decisions - 1
    rows = len(decisions)
    names = len(ids)
    components = np.full((rows, names, 4), np.nan, dtype=np.float32)
    volatility = np.full((rows, names), np.nan, dtype=np.float32)
    context = np.full((rows, names, 3), np.nan, dtype=np.float32)
    original_profile = np.full((rows, names), 255, dtype=np.uint8)
    finite = np.zeros((rows, names), dtype=bool)
    for column in range(names):
        observed = np.flatnonzero(np.isfinite(close[:, column]) & tradable[:, column])
        if len(observed) < 520:
            continue
        prices = close[observed, column].astype(np.float64)
        returns = np.full(len(prices), np.nan, dtype=np.float64)
        returns[1:] = prices[1:] / prices[:-1] - 1.0
        factor, state, ready, profile, _ = copied_stockalgo_signals(prices, returns)
        bars = np.searchsorted(observed, signal_dates, side="right") - 1
        valid = (bars >= 0)
        valid[valid] &= ready[bars[valid]]
        row_indices = np.flatnonzero(valid)
        selected_bars = bars[valid]
        components[row_indices, column] = factor[selected_bars]
        volatility[row_indices, column] = np.maximum(0.05, state[selected_bars, 4] * 0.05)
        context[row_indices, column, 0] = state[selected_bars, 9]
        context[row_indices, column, 1] = state[selected_bars, 10] * 0.05
        context[row_indices, column, 2] = state[selected_bars, 11]
        original_profile[row_indices, column] = profile[selected_bars]
        finite[row_indices, column] = True
        if (column + 1) % 100 == 0 or column + 1 == names:
            print(f"Reconstructed ensemble signals {column + 1:,}/{names:,}", flush=True)
    return components, volatility, context, original_profile, finite


def market_context(benchmark, dates, decisions, eligible, tradable, components, context, finite):
    rows = []
    for row_index, decision in enumerate(decisions):
        signal_date = decision - 1
        active = eligible[decision] & tradable[decision] & finite[row_index]
        history = benchmark[:signal_date]
        values = []
        for window in (5, 21, 63, 252):
            if len(history) >= window:
                values.append(float(np.prod(1 + history[-window:]) - 1))
            else:
                values.append(0.0)
        for window in (21, 63):
            values.append(float(np.std(history[-window:], ddof=1)) if len(history) >= window else 0.0)
        if len(history):
            curve = np.cumprod(1 + history)
            values.append(float(curve[-1] / np.max(curve[-252:]) - 1) if len(curve) >= 2 else 0.0)
        else:
            values.append(0.0)
        if active.any():
            factors = components[row_index, active]
            conditions = context[row_index, active]
            means = np.nanmean(factors, axis=0)
            values.extend([
                float(np.clip(means[0] / 0.1, -5, 5)),
                float(np.clip(means[1] / 0.02, -5, 5)),
                float(np.clip(means[2] / 0.2, -5, 5)),
                float(np.mean(conditions[:, 0] > 0.7)),
                float(np.clip(np.nanmean(conditions[:, 1]) / 0.05, 0, 5)),
                float(np.mean(conditions[:, 2] > 0)),
                float(np.mean(factors[:, 0] > 0)),
                float(np.mean(factors[:, 1] > 0)),
                float(np.mean(factors[:, 2] > 0)),
                float(np.log1p(active.sum()) / np.log(1000)),
            ])
        else:
            values.extend([0.0] * 10)
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def policy_weights(action, original_profile, index, eligible, tradable, held, finite):
    n = held.shape[0]
    available = tradable[index] & (eligible[index] | held) & finite
    locked = held & ~tradable[index]
    if isinstance(action, str) and action == "source":
        profiles = original_profile
    elif not np.isscalar(action):
        profiles = np.asarray(action, dtype=np.uint8)
    else:
        profiles = np.full(n, int(action), dtype=np.uint8)
    return available, locked, profiles


def rebalance(index, row_index, action, current_weights, components, volatility,
              original_profile, eligible, tradable, finite):
    available, locked, profiles = policy_weights(
        action, original_profile[row_index], index, eligible, tradable,
        current_weights > 0, finite[row_index],
    )
    target = np.zeros_like(current_weights)
    target[locked] = current_weights[locked]
    for profile in range(len(WEIGHT_PROFILES)):
        members = np.flatnonzero(available & (profiles == profile))
        if not len(members):
            continue
        scores = components[row_index, members] @ WEIGHT_PROFILES[profile]
        for offset, column in enumerate(members):
            score = float(scores[offset])
            if score > 0.02:
                size = min(0.10, 0.15 / max(float(volatility[row_index, column]), 0.05))
                target[column] = size
            elif score < -0.02:
                target[column] = 0.0
            elif current_weights[column] > 0:
                target[column] = current_weights[column]

    available_budget = max(0.0, 1.0 - float(target[locked].sum()))
    tradable_total = float(target[available].sum())
    if tradable_total > available_budget and tradable_total > 0:
        target[available] *= available_budget / tradable_total
    turnover = float(np.abs(target - current_weights).sum())
    counts = np.bincount(profiles[available].astype(np.int64), minlength=len(WEIGHT_PROFILES))
    profile_mix = counts / max(int(counts.sum()), 1)
    return target, max(0.0, 1.0 - float(target.sum())), turnover, profile_mix


def hold_period(start, target, cash, returns, cash_returns, settlements, assumed, days):
    weights = target.copy()
    current_cash = float(cash)
    total_growth = 1.0
    assumed_exposure_days = 0.0
    for day in range(start, start + days):
        held = weights > 0
        row_returns = returns[day]
        if np.any(held & ~np.isfinite(row_returns)):
            raise ValueError(f"Nonfinite held return on row {day}")
        growth_values = weights.copy()
        growth_values[held] *= 1.0 + row_returns[held]
        current_cash *= 1.0 + float(cash_returns[day])
        settle = settlements[day] & (growth_values > 0)
        current_cash += float(growth_values[settle].sum())
        growth_values[settle] = 0.0
        growth = float(growth_values.sum() + current_cash)
        if not np.isfinite(growth) or growth <= 0:
            raise ValueError(f"Invalid portfolio value on row {day}")
        assumed_exposure_days += float(weights[assumed[day]].sum())
        weights = growth_values / growth
        current_cash /= growth
        total_growth *= growth
    return weights, current_cash, total_growth, assumed_exposure_days / days


def global_state(base, cash):
    return np.concatenate([base, [cash]]).astype(np.float32)


def simulate_action(action, index, row_index, weights, cash, components, volatility,
                    original_profile, eligible, tradable, finite, returns,
                    cash_returns, settlements, assumed, commission_bps=10):
    target, cash_target, turnover, profile_mix = rebalance(
        index, row_index, action, weights, components, volatility,
        original_profile, eligible, tradable, finite,
    )
    cost = commission_bps / 10000 * turnover
    # Commissions reduce equity pro rata while preserving target proportions.
    after_cost = max(1e-8, 1.0 - cost)
    target *= after_cost
    cash_target *= after_cost
    next_weights, next_cash, gross, assumed_exposure = hold_period(
        index, target, cash_target, returns, cash_returns, settlements, assumed, INTERVAL,
    )
    net_growth = gross
    reward = float(np.log(max(net_growth, 1e-8)) * 100)
    record = {
        "turnover": turnover,
        "cost_fraction": cost,
        "period_return": net_growth - 1.0,
        "cash_weight": next_cash,
        "assumed_terminal_exposure": assumed_exposure,
        "invested_weight": float(next_weights.sum()),
        "mean_reversion_profile_share": float(profile_mix[0]),
        "momentum_profile_share": float(profile_mix[1]),
        "balanced_profile_share": float(profile_mix[2]),
    }
    return next_weights, next_cash, reward, record


def train_agent(states, decisions, decision_rows, components, volatility, original_profile,
                finite, eligible, tradable, returns, cash_returns, settlements, assumed,
                bounds, episodes, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    left, right = bounds
    indices = [idx for idx in decisions if left <= idx < right - INTERVAL]
    rows_by_index = {idx: decision_rows[idx] for idx in indices}
    model = WeightQNetwork(states.shape[1] + 1, len(WEIGHT_PROFILES))
    target = WeightQNetwork(states.shape[1] + 1, len(WEIGHT_PROFILES))
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    replay = deque(maxlen=40_000)
    epsilon = 1.0
    total_steps = 0
    training = []

    for episode in range(episodes):
        length = rng.randint(20, min(50, len(indices) - 1))
        start = rng.randint(0, len(indices) - length - 1)
        weights = np.zeros(eligible.shape[1], dtype=np.float64)
        cash = 1.0
        episode_reward = 0.0
        for step in range(length):
            index = indices[start + step]
            row_index = rows_by_index[index]
            state = global_state(states[row_index], cash)
            if rng.random() < epsilon:
                action = rng.randrange(len(WEIGHT_PROFILES))
            else:
                with torch.no_grad():
                    action = int(model(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
            next_weights, next_cash, reward, _ = simulate_action(
                action, index, row_index, weights, cash, components, volatility,
                original_profile, eligible, tradable, finite, returns, cash_returns,
                settlements, assumed,
            )
            next_index = index + INTERVAL
            next_row = rows_by_index.get(next_index)
            if next_row is None:
                next_state = state
                done = True
            else:
                next_state = global_state(states[next_row], next_cash)
                done = step + 1 == length
            replay.append((state, action, reward, next_state, done))
            weights, cash = next_weights, next_cash
            episode_reward += reward
            total_steps += 1
            epsilon = max(0.05, epsilon * 0.9997)
            if len(replay) >= 128:
                update_network(model, target, optimizer, rng.sample(replay, 128), 0.97)
            if total_steps % 500 == 0:
                target.load_state_dict(model.state_dict())
        training.append({"episode": episode + 1, "reward": episode_reward, "epsilon": epsilon})
        if (episode + 1) % 25 == 0:
            target.load_state_dict(model.state_dict())

    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "adaptive_universe_weights_dqn.pt")
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(training)
    return model


def stock_observation(row, column, cash, prior_exposure, states, components, context):
    scaled_components = np.asarray([
        np.clip(components[row, column, 0] / 0.1, -5, 5),
        np.clip(components[row, column, 1] / 0.02, -5, 5),
        np.clip(components[row, column, 2] / 0.2, -5, 5),
        components[row, column, 3],
    ], dtype=np.float32)
    return np.concatenate([
        scaled_components,
        context[row, column],
        states[row],
        np.asarray([cash, prior_exposure], dtype=np.float32),
    ]).astype(np.float32)


def train_per_security_agent(states, decisions, decision_rows, components, context,
                             finite, eligible, tradable, returns, episodes, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = WeightQNetwork(states.shape[1] + 4 + 3 + 2, len(WEIGHT_PROFILES))
    target = WeightQNetwork(states.shape[1] + 4 + 3 + 2, len(WEIGHT_PROFILES))
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    replay = deque(maxlen=50_000)
    train_indices = [idx for idx in decisions if idx + INTERVAL < eligible.shape[0]]
    epsilon = 1.0
    total_steps = 0
    history = []

    for episode in range(episodes):
        start_pos = rng.randrange(max(1, len(train_indices) - 60))
        start_index = train_indices[start_pos]
        row = decision_rows[start_index]
        candidates = np.flatnonzero(finite[row] & eligible[start_index] & tradable[start_index])
        if not len(candidates):
            continue
        column = int(rng.choice(candidates.tolist()))
        length = rng.randint(20, 50)
        prior_exposure = 0.0
        episode_reward = 0.0
        for step in range(length):
            pos = start_pos + step
            if pos + 1 >= len(train_indices):
                break
            index = train_indices[pos]
            next_index = train_indices[pos + 1]
            if next_index != index + INTERVAL:
                break
            row = decision_rows[index]
            next_row = decision_rows[next_index]
            if not (finite[row, column] and eligible[index, column] and tradable[index, column]):
                break
            period_returns = np.asarray(returns[index : index + INTERVAL, column])
            if not np.isfinite(period_returns).all():
                break
            state = stock_observation(row, column, 1.0 - prior_exposure, prior_exposure, states, components, context)
            if rng.random() < epsilon:
                action = rng.randrange(len(WEIGHT_PROFILES))
            else:
                with torch.no_grad():
                    action = int(model(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
            score = float(components[row, column] @ WEIGHT_PROFILES[action])
            exposure = 1.0 if score > 0.02 else 0.0 if score < -0.02 else prior_exposure
            period_return = float(np.prod(1.0 + exposure * period_returns) - 1.0)
            cost = 0.001 * abs(exposure - prior_exposure)
            reward = float(np.log(max(1e-8, (1.0 + period_return) * (1.0 - cost))) * 100)
            next_state = stock_observation(next_row, column, 1.0 - exposure, exposure, states, components, context)
            done = step + 1 == length
            replay.append((state, action, reward, next_state, done))
            prior_exposure = exposure
            episode_reward += reward
            total_steps += 1
            epsilon = max(0.05, epsilon * 0.9997)
            if len(replay) >= 128:
                update_network(model, target, optimizer, rng.sample(replay, 128), 0.97)
            if total_steps % 500 == 0:
                target.load_state_dict(model.state_dict())
        history.append({"episode": episode + 1, "reward": episode_reward, "epsilon": epsilon})
        if (episode + 1) % 25 == 0:
            target.load_state_dict(model.state_dict())

    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "adaptive_universe_per_security_dqn.pt")
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(history)
    return model


def evaluate(policy, model, dates, decisions, decision_rows, states, components, volatility,
             context, original_profile, finite, eligible, tradable, returns, cash_returns,
             settlements, assumed, bounds, agent_type="global"):
    left, right = bounds
    indices = [idx for idx in decisions if left <= idx < right - INTERVAL]
    weights = np.zeros(eligible.shape[1], dtype=np.float64)
    cash = 1.0
    wealth = 1.0
    peak = 1.0
    records = []
    for index in indices:
        row_index = decision_rows[index]
        state = global_state(states[row_index], cash)
        if policy == "rl_weights":
            if agent_type == "per_security":
                available = finite[row_index] & tradable[index] & (eligible[index] | (weights > 0))
                columns = np.flatnonzero(available)
                action = np.full(len(weights), 2, dtype=np.uint8)
                if len(columns):
                    stock_states = np.asarray([
                        stock_observation(row_index, column, cash, weights[column], states, components, context)
                        for column in columns
                    ])
                    with torch.no_grad():
                        action[columns] = model(torch.from_numpy(stock_states)).argmax(dim=1).numpy().astype(np.uint8)
            else:
                with torch.no_grad():
                    action = int(model(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
        elif policy == "stockalgo_original":
            action = "source"
        else:
            action = 2
        next_weights, next_cash, reward, info = simulate_action(
            action, index, row_index, weights, cash, components, volatility,
            original_profile, eligible, tradable, finite, returns, cash_returns,
            settlements, assumed,
        )
        wealth *= 1.0 + info["period_return"]
        peak = max(peak, wealth)
        info.update({
            "date": str(dates[index]),
            "policy": policy,
            "weight_profile": (
                "source" if isinstance(action, str) else
                "adaptive_per_security" if not np.isscalar(action) else
                PROFILE_NAMES[int(action)]
            ),
            "action": -1 if isinstance(action, str) else "per_security" if not np.isscalar(action) else int(action),
            "wealth": wealth,
            "drawdown": wealth / peak - 1,
            "reward": reward,
        })
        records.append(info)
        weights, cash = next_weights, next_cash
    return records


def summarize(records):
    returns = np.asarray([row["period_return"] for row in records], dtype=np.float64)
    wealth = np.asarray([row["wealth"] for row in records], dtype=np.float64)
    years = len(records) * INTERVAL / 252
    std = float(np.std(returns, ddof=1))
    annualizer = np.sqrt(252 / INTERVAL)
    return {
        "intervals": len(records),
        "total_return_pct": float((wealth[-1] - 1) * 100),
        "cagr_pct": float((wealth[-1] ** (1 / years) - 1) * 100),
        "annualized_volatility_pct": std * annualizer * 100,
        "sharpe_zero_rate": float(np.mean(returns) / std * annualizer) if std > 0 else 0.0,
        "max_drawdown_pct": float(min(row["drawdown"] for row in records) * 100),
        "mean_cash_pct": float(np.mean([row["cash_weight"] for row in records]) * 100),
        "mean_invested_weight_pct": float(np.mean([row["invested_weight"] for row in records]) * 100),
        "annual_turnover": float(sum(row["turnover"] for row in records) / years),
        "mean_assumed_terminal_exposure_pct": float(np.mean([row["assumed_terminal_exposure"] for row in records]) * 100),
        "weight_profile_pct": {
            name: float(np.mean([row["weight_profile"] == name for row in records]) * 100)
            for name in (*PROFILE_NAMES, "source", "adaptive_per_security")
        },
        "mean_profile_mix_pct": {
            name: float(np.mean([row[column] for row in records]) * 100)
            for name, column in (
                ("mean_reversion", "mean_reversion_profile_share"),
                ("momentum", "momentum_profile_share"),
                ("balanced", "balanced_profile_share"),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description="RL adapts copied StockAlgo ensemble weights on historical S&P constituents.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--agent", choices=("global", "per_security"), default="global")
    parser.add_argument("--output", type=Path, default=Path("outputs/adaptive-ensemble-universe-dqn-001"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--staging", type=Path, default=STAGING)
    parser.add_argument("--research", type=Path, default=RESEARCH)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--membership", type=Path, default=MEMBERSHIP)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    dates = np.load(args.staging / "dates.npy").astype("datetime64[D]")
    ids = np.load(args.staging / "security_ids.npy").astype(str)
    eligible = np.load(args.staging / "eligible.npy")
    tradable = np.load(args.staging / "tradable.npy")
    returns = np.load(args.research / "returns.npy", mmap_mode="r")
    cash_returns = np.load(args.staging / "cash_returns.npy", mmap_mode="r")
    settlements = np.load(args.research / "cash_settlements.npy", mmap_mode="r")
    assumed = np.load(args.research / "assumed_exits.npy", mmap_mode="r")
    benchmark = np.load(args.staging / "benchmark.npy", mmap_mode="r")

    train_bounds = dated_bounds(dates, "1996-01-01", "2012-12-31")
    validation_bounds = dated_bounds(dates, "2013-01-01", "2018-12-31")
    test_bounds = dated_bounds(dates, "2019-01-01", str(dates[-1]))
    decisions = np.unique(np.concatenate([
        np.arange(left, right - INTERVAL, INTERVAL)
        for left, right in (train_bounds, validation_bounds, test_bounds)
    ]))
    decision_rows = {int(index): row for row, index in enumerate(decisions)}

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output / "signal_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_names = ("components", "volatility", "context", "original_profile", "finite")
    cache_meta = cache_dir / "manifest.json"
    expected_cache = {
        "price_source": str(args.prices.resolve()),
        "dates": [str(dates[0]), str(dates[-1])],
        "decision_count": int(len(decisions)),
        "security_count": int(len(ids)),
        "signal_timing": "signal dates are one session before the rebalance date",
    }
    if cache_meta.exists() and json.loads(cache_meta.read_text()) == expected_cache and all((cache_dir / f"{name}.npy").exists() for name in cache_names):
        cached = [np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in cache_names]
        components, volatility, context, original_profile, finite = cached
        print("Reusing the saved signal panel", flush=True)
    else:
        close = load_close_panel(args.prices, args.membership, dates, ids, tradable)
        components, volatility, context, original_profile, finite = build_signal_panel(
            close, dates, ids, decisions, tradable,
        )
        del close
        for name, values in zip(cache_names, (components, volatility, context, original_profile, finite)):
            np.save(cache_dir / f"{name}.npy", values)
        cache_meta.write_text(json.dumps(expected_cache, indent=2) + "\n")
    states = market_context(benchmark, dates, decisions, eligible, tradable, components, context, finite)
    if args.agent == "global":
        del context
    trainer = train_agent if args.agent == "global" else train_per_security_agent
    if args.agent == "global":
        network = trainer(
            states, decisions, decision_rows, components, volatility, original_profile,
            finite, eligible, tradable, returns, cash_returns, settlements, assumed,
            train_bounds, args.episodes, args.seed, args.output,
        )
    else:
        network = trainer(
            states, decisions, decision_rows, components, context, finite, eligible,
            tradable, returns, args.episodes, args.seed, args.output,
        )
    records = []
    for split, bounds in (("validation", validation_bounds), ("test", test_bounds)):
        for policy in ("rl_weights", "stockalgo_original", "balanced_weights"):
            records.extend(evaluate(
                policy, network, dates, decisions, decision_rows, states, components,
                volatility, context, original_profile, finite, eligible, tradable,
                returns, cash_returns, settlements, assumed, bounds, args.agent,
            ))
    with (args.output / "weekly_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = {
        split: {
            policy: summarize([row for row in records if row["date"] >= str(dates[bounds[0]]) and row["date"] <= str(dates[bounds[1] - 1]) and row["policy"] == policy])
            for policy in ("rl_weights", "stockalgo_original", "balanced_weights")
        }
        for split, bounds in (("validation", validation_bounds), ("test", test_bounds))
    }
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    assumptions = json.loads((args.research / "manifest.json").read_text())["research_assumptions"]
    (args.output / "run.json").write_text(json.dumps({
        "algorithm": (
            "Double DQN selects one weight profile per week for the whole historical constituent portfolio"
            if args.agent == "global"
            else "Double DQN selects weights separately for each eligible constituent every week using a shared per-stock policy"
        ),
        "agent": args.agent,
        "source_copy": "rl_pm.adaptive_ensemble_rl.copied_stockalgo_signals; StockAlgo original was not changed",
        "weight_actions": {name: weights.tolist() for name, weights in zip(PROFILE_NAMES, WEIGHT_PROFILES)},
        "trade_rule": "source score threshold and long-only BUY/HOLD/SELL behavior; 10% per-security cap and 15% volatility target; 10 bps commission",
        "execution_alignment": "signal computed from the last available close before each rebalance date; forward returns begin at that rebalance date",
        "training_period": [str(dates[train_bounds[0]]), str(dates[train_bounds[1] - 1])],
        "validation_period": [str(dates[validation_bounds[0]]), str(dates[validation_bounds[1] - 1])],
        "test_period": [str(dates[test_bounds[0]]), str(dates[test_bounds[1] - 1])],
        "episodes": args.episodes,
        "seed": args.seed,
        "research_assumptions": assumptions,
        "limitations": ["historical identity/membership and terminal returns are not fully verified", "unknown delisted exits use zero-recovery stress assumptions", "global weight decision is shared across securities", "one seed"],
    }, indent=2) + "\n")

    test_start = str(dates[test_bounds[0]])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for policy in ("rl_weights", "stockalgo_original", "balanced_weights"):
        rows = [row for row in records if row["date"] >= test_start and row["policy"] == policy]
        axes[0].plot([np.datetime64(row["date"]) for row in rows], [row["wealth"] for row in rows], label=policy.replace("_", " "))
    axes[0].set_title("Historical constituent portfolio — held-out test")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    rows = [row for row in records if row["date"] >= test_start and row["policy"] == "rl_weights"]
    for name, column in (
        ("mean reversion", "mean_reversion_profile_share"),
        ("momentum", "momentum_profile_share"),
        ("balanced", "balanced_profile_share"),
    ):
        axes[1].step([np.datetime64(row["date"]) for row in rows], [row[column] for row in rows], where="post", label=name)
    axes[1].set_title("Constituent-level weight-profile mix selected by RL")
    axes[1].set_ylabel("Share of tradable members")
    axes[1].set_ylim(0, 1)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "adaptive_weights_universe_test.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))
    print(f"Results: {args.output / 'weekly_results.csv'}")


if __name__ == "__main__":
    main()
