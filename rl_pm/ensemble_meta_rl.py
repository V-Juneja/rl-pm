from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_pm.index_exposure_rl import (
    ACTIONS,
    PRICE_FILE,
    dated_bounds,
    load_series,
    train,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "market_return_1d",
    "market_return_5d",
    "market_return_21d",
    "market_return_63d",
    "market_vol_21d",
    "market_vol_63d",
    "market_drawdown_126d",
    "market_sma_50_distance",
    "market_sma_200_distance",
    "ensemble_score_scaled",
    "ensemble_price_momentum",
    "ensemble_earnings_proxy",
    "ensemble_mean_reversion",
    "ensemble_rsi_reversion",
    "ensemble_value_score",
    "ensemble_low_volatility",
    "ensemble_volatility_regime",
    "ensemble_trend_strength",
    "ensemble_market_stress",
    "walk_forward_knn_signal",
    "walk_forward_knn_confidence",
)


def rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(window - 1, len(values)):
        current = values[index]
        sample = values[index - window + 1 : index + 1]
        less = np.sum(sample < current)
        tied = np.sum(sample == current)
        result[index] = (less + (tied + 1) / 2) / window
    return result


def stockalgo_features(close: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    count = len(close)

    def mean(values, window):
        return np.convolve(values, np.ones(window) / window, mode="full")[:count]

    def rolling_std(values, window):
        result = np.full(count, np.nan)
        for index in range(window - 1, count):
            result[index] = np.std(values[index - window + 1 : index + 1], ddof=1)
        return result

    mom_21 = close / np.roll(close, 20) - 1.0
    mom_63 = close / np.roll(close, 62) - 1.0
    mom_126 = close / np.roll(close, 125) - 1.0
    mom_21[:20] = np.nan
    mom_63[:62] = np.nan
    mom_126[:125] = np.nan

    vol_20 = rolling_std(returns, 20) * np.sqrt(252)
    vol_126 = rolling_std(returns, 126) * np.sqrt(252)
    vol_rank = rolling_percentile(vol_20, 500)

    log_returns = np.zeros(count)
    log_returns[1:] = np.log1p(returns[1:])
    recent_mean = mean(log_returns, 20)
    mean_reversion = -(log_returns - recent_mean)

    delta = np.diff(close, prepend=close[0])
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = mean(gains, 14)
    avg_loss = mean(losses, 14)
    rs = np.divide(avg_gain, avg_loss, out=np.full(count, np.inf), where=avg_loss > 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_reversion = (50 - rsi) / 50

    sma_20 = mean(close, 20)
    sma_50 = mean(close, 50)
    sma_200 = mean(close, 200)
    std_20 = rolling_std(close, 20)
    band_width = 4 * std_20
    bb_position = np.divide(
        close - (sma_20 - 2 * std_20),
        band_width,
        out=np.full(count, 0.5),
        where=band_width > 0,
    )
    value_50 = close / sma_50 - 1.0
    value_200 = close / sma_200 - 1.0
    bb_value = 1 - bb_position
    value_score = np.mean(np.column_stack([value_50, value_200, bb_value]), axis=1)

    trend_strength = np.abs(sma_20 - sma_50) / np.maximum(sma_50, 1e-12)
    market_stress = ((rsi > 70) | (rsi < 30)).astype(np.float64)
    price_momentum = np.divide(
        mom_126,
        np.where(np.isfinite(vol_126) & (vol_126 > 0), vol_126, 0.2),
    )
    earnings_proxy = mom_21 - mom_63

    momentum_weight = np.full(count, 0.25)
    mean_reversion_weight = np.full(count, 0.25)
    value_weight = np.full(count, 0.25)
    high_vol = vol_rank > 0.7
    strong_trend = (~high_vol) & (trend_strength > 0.05)
    momentum_weight[high_vol] = 0.2
    mean_reversion_weight[high_vol] = 0.4
    value_weight[high_vol] = 0.2
    momentum_weight[strong_trend] = 0.4
    mean_reversion_weight[strong_trend] = 0.2
    value_weight[strong_trend] = 0.2

    momentum_score = 0.6 * price_momentum + 0.4 * earnings_proxy
    reversion_score = 0.6 * mean_reversion + 0.4 * rsi_reversion
    # ensemble.py currently asks for "knn_prediction", while its KNN method
    # returns "ml_signal". Preserve the source score's zero ML contribution.
    ensemble_score = (
        momentum_score * momentum_weight
        + reversion_score * mean_reversion_weight
        + value_score * value_weight
    )

    features = np.column_stack(
        [
            price_momentum,
            earnings_proxy / 0.1,
            mean_reversion / 0.02,
            rsi_reversion,
            value_score / 0.2,
            1 - vol_rank,
            vol_rank,
            trend_strength / 0.05,
            market_stress,
        ]
    )
    features[:, 0] = np.clip(features[:, 0], -5, 5)
    features[:, 1:5] = np.clip(features[:, 1:5], -5, 5)
    features[:, 5:7] = np.clip(features[:, 5:7], 0, 1)
    features[:, 7] = np.clip(features[:, 7], 0, 5)
    features = np.column_stack(
        [
            # Same trailing market indicators as the original SPY exposure RL.
            np.clip(returns / 0.03, -5, 5),
            np.clip((close / np.roll(close, 5) - 1) / 0.06, -5, 5),
            np.clip(mom_21 / 0.12, -5, 5),
            np.clip(mom_63 / 0.20, -5, 5),
            np.clip(vol_20 / 0.05, 0, 5),
            np.clip(vol_126 / 0.05, 0, 5),
            np.clip(close / np.maximum.accumulate(close) - 1, -5, 0),
            np.clip(value_50 / 0.15, -5, 5),
            np.clip(value_200 / 0.25, -5, 5),
            np.clip(ensemble_score / 0.02, -5, 5),
            features,
        ]
    ).astype(np.float32)
    features[0, :4] = np.nan
    features[:125, 2] = np.nan
    features[:62, 3] = np.nan
    features[:20, 4] = np.nan
    features[:126, 5] = np.nan
    features[:200, 8] = np.nan
    features[:499, 15:17] = np.nan
    return features, ensemble_score.astype(np.float32)


def causal_knn_signal(features: np.ndarray, returns: np.ndarray, lookback: int = 1000, neighbors: int = 5):
    signal = np.full(len(features), np.nan, dtype=np.float32)
    confidence = np.full(len(features), np.nan, dtype=np.float32)
    first = int(np.flatnonzero(np.isfinite(features).all(axis=1))[0])
    for index in range(first + neighbors, len(features)):
        start = max(first, index - lookback)
        train_x = features[start:index]
        train_y = np.sign(returns[start + 1 : index + 1])
        valid = np.isfinite(train_x).all(axis=1) & np.isfinite(train_y) & (train_y != 0)
        train_x = train_x[valid]
        train_y = train_y[valid]
        if len(train_y) < neighbors:
            continue
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        scaled_train = (train_x - mean) / scale
        scaled_query = (features[index] - mean) / scale
        distances = np.sum((scaled_train - scaled_query) ** 2, axis=1)
        nearest = np.argpartition(distances, neighbors - 1)[:neighbors]
        estimate = float(np.mean(train_y[nearest]))
        signal[index] = estimate
        confidence[index] = abs(estimate)
    return signal, confidence


def policy_exposure(policy, network, state, ensemble_score, momentum_126):
    if policy == "dqn_meta":
        with torch.no_grad():
            q_values = network(torch.tensor(state).unsqueeze(0))
        return float(ACTIONS[int(q_values.argmax(dim=1).item())])
    if policy == "stockalgo_buy_sell":
        return 1.0 if ensemble_score > 0.02 else 0.0
    if policy == "stockalgo_linear":
        return float(np.clip(0.5 + ensemble_score / 0.04 * 0.5, 0, 1))
    if policy == "stockalgo_plus_knn":
        return 1.0 if ensemble_score > 0.02 else 0.0
    if policy == "momentum_126":
        return 1.0 if momentum_126 > 0 else 0.0
    if policy == "cash":
        return 0.0
    if policy == "buy_hold":
        return 1.0
    raise ValueError(policy)


def evaluate(dates, returns, close, features, ensemble_score, knn_signal, bounds, network, split):
    left, right = bounds
    policies = (
        "dqn_meta",
        "stockalgo_buy_sell",
        "stockalgo_linear",
        "stockalgo_plus_knn",
        "momentum_126",
        "buy_hold",
        "cash",
    )
    records = []
    for policy in policies:
        wealth = 1.0
        exposure = 0.0
        peak = 1.0
        for index in range(left, right - 1, 5):
            state = np.concatenate([features[index], np.asarray([exposure], dtype=np.float32)])
            score = float(ensemble_score[index])
            effective_score = score
            if policy == "stockalgo_plus_knn":
                effective_score += float(knn_signal[index]) * 0.2
            momentum_126 = close[index] / close[index - 125] - 1 if index >= 125 else 0
            target_exposure = policy_exposure(
                policy, network, state.astype(np.float32), effective_score, momentum_126
            )
            next_index = min(index + 5, right - 1)
            period_days = next_index - index
            period_return = np.prod(
                1.0 + target_exposure * returns[index + 1 : next_index + 1]
            ) - 1.0
            turnover_cost = 0.0005 * abs(target_exposure - exposure)
            period_growth = (1.0 + period_return) * max(1e-8, 1.0 - turnover_cost)
            period_net_return = period_growth - 1.0
            wealth *= period_growth
            peak = max(peak, wealth)
            records.append(
                {
                    "split": split,
                    "date": str(dates[index + 1]),
                    "policy": policy,
                    "ensemble_score": effective_score,
                    "exposure": target_exposure,
                    "period_return": period_net_return,
                    "period_days": period_days,
                    "turnover": abs(target_exposure - exposure),
                    "wealth": wealth,
                    "drawdown": wealth / peak - 1,
                }
            )
            exposure = target_exposure
    return records


def summarize_weekly(records):
    report = {}
    for policy in sorted({row["policy"] for row in records}):
        rows = [row for row in records if row["policy"] == policy]
        period_returns = np.asarray([row["period_return"] for row in rows], dtype=np.float64)
        days = np.asarray([row["period_days"] for row in rows], dtype=np.float64)
        wealth = np.asarray([row["wealth"] for row in rows], dtype=np.float64)
        years = float(days.sum() / 252.0)
        annualization = np.sqrt(252.0 / float(np.mean(days)))
        period_std = float(np.std(period_returns, ddof=1)) if len(rows) > 1 else 0.0
        cagr = float(wealth[-1] ** (1 / years) - 1) if years > 0 and wealth[-1] > 0 else -1.0
        report[policy] = {
            "sessions": int(days.sum()),
            "total_return_pct": float((wealth[-1] - 1.0) * 100),
            "cagr_pct": cagr * 100,
            "annualized_volatility_pct": period_std * annualization * 100,
            "sharpe_zero_rate": float(np.mean(period_returns) / period_std * annualization) if period_std > 0 else 0.0,
            "max_drawdown_pct": float(min(row["drawdown"] for row in rows) * 100),
            "mean_exposure_pct": float(np.mean([row["exposure"] for row in rows]) * 100),
            "annual_turnover": float(sum(row["turnover"] for row in rows) / years) if years else 0.0,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Train a DQN meta-policy over StockAlgo ensemble signals on historical SPY.")
    parser.add_argument("--prices", type=Path, default=PRICE_FILE)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path, default=Path("outputs/ensemble-meta-dqn-001"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    dates, close, returns, _ = load_series(args.prices)
    features, ensemble_score = stockalgo_features(close, returns)
    knn_signal, knn_confidence = causal_knn_signal(features, returns)
    features = np.column_stack([features, knn_signal, knn_confidence]).astype(np.float32)
    valid = np.isfinite(features).all(axis=1)
    train_bounds = dated_bounds(dates, "1996-01-01", "2012-12-31")
    validation_bounds = dated_bounds(dates, "2013-01-01", "2018-12-31")
    test_bounds = dated_bounds(dates, "2019-01-01", str(dates[-1]))
    first_valid = int(np.flatnonzero(valid)[0])
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
        interval=5,
    )
    records = []
    records.extend(evaluate(dates, returns, close, features, ensemble_score, knn_signal, validation_bounds, network, "validation"))
    records.extend(evaluate(dates, returns, close, features, ensemble_score, knn_signal, test_bounds, network, "test"))
    with (args.output / "weekly_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = {
        split: summarize_weekly([row for row in records if row["split"] == split])
        for split in ("validation", "test")
    }
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "features.csv").write_text(
        "date,ensemble_score," + ",".join(FEATURE_NAMES) + "\n"
    )
    with (args.output / "features.csv").open("a", newline="") as stream:
        writer = csv.writer(stream)
        for index, date in enumerate(dates):
            writer.writerow([str(date), float(ensemble_score[index]), *features[index].tolist()])

    metadata = {
        "algorithm": "Double DQN meta-policy; exposure actions [0, 0.5, 1]; decisions every five sessions",
        "base_signal_formula_source": "StockAlgo ensemble.py signal formulas mirrored in this module",
        "base_signal_timing": "close t features only; next five sessions of adjusted returns are reward",
        "base_signal_ml_contribution": "zero, matching current ensemble.py output key behavior (ml_signal vs knn_prediction)",
        "meta_ml_signal": "causal 5-neighbor classifier over prior SPY signal states; trains only on next-session labels observed by the current close",
        "transaction_cost_bps_per_unit_turnover": 5,
        "training_period": [str(dates[train_bounds[0]]), str(dates[train_bounds[1] - 1])],
        "validation_period": [str(dates[validation_bounds[0]]), str(dates[validation_bounds[1] - 1])],
        "test_period": [str(dates[test_bounds[0]]), str(dates[test_bounds[1] - 1])],
        "training_episodes": args.episodes,
        "seed": args.seed,
        "features": list(FEATURE_NAMES),
        "price_source": str(args.prices.resolve().relative_to(ROOT)) if args.prices.resolve().is_relative_to(ROOT) else args.prices.name,
        "limitations": [
            "applies StockAlgo single-name factor formulas to SPY; it does not replay historical email picks",
            "SPY data starts in 1995, shorter than the reconstructed constituent panel",
            "cash earns zero; no risk-free yield or taxes",
            "one held-out period; results are research, not a live-trading claim",
        ],
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    test = [row for row in records if row["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    for policy in sorted({row["policy"] for row in test}):
        rows = [row for row in test if row["policy"] == policy]
        axes[0].plot([np.datetime64(row["date"]) for row in rows], [row["wealth"] for row in rows], label=policy, linewidth=1.2)
    axes[0].set_title("Untouched test period")
    axes[0].set_ylabel("Growth of $1")
    axes[0].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25)
    rows = [row for row in test if row["policy"] == "dqn_meta"]
    axes[1].step([np.datetime64(row["date"]) for row in rows], [row["exposure"] for row in rows], color="#5B8E7D", where="post")
    axes[1].set_title("DQN meta-policy exposure")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "meta_exposure_test.png", dpi=170, facecolor="white")
    plt.close(fig)

    print(json.dumps(summaries, indent=2))
    print(f"Results: {args.output / 'weekly_results.csv'}")
    print(f"Chart: {args.output / 'meta_exposure_test.png'}")


if __name__ == "__main__":
    main()
