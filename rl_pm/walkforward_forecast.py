from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from rl_pm.ensemble_meta_rl import stockalgo_features
from rl_pm.index_exposure_rl import PRICE_FILE, dated_bounds, load_series


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
)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 25.0):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (x - mean) / scale
    target_mean = float(y.mean())
    centered_y = y - target_mean
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), standardized.T @ centered_y)
    return mean, scale, target_mean, coefficients


def forecasts_walk_forward(features, target, valid, start, stop, first_train=0, refit_every=21, lookback=2520):
    predictions = np.full(len(features), np.nan, dtype=np.float64)
    current_model = None
    last_fit_index = -refit_every
    for index in range(start, stop - 1, 5):
        if current_model is None or index - last_fit_index >= refit_every:
            # A five-session label is usable only after all five returns have closed.
            train_end = index - 4
            train_start = max(first_train, train_end - lookback)
            rows = np.arange(train_start, train_end)
            usable = valid[rows] & np.isfinite(target[rows])
            rows = rows[usable]
            if len(rows) < 500:
                continue
            current_model = fit_ridge(features[rows], target[rows])
            last_fit_index = index
        mean, scale, target_mean, coefficients = current_model
        z = (features[index] - mean) / scale
        predictions[index] = target_mean + float(z @ coefficients)
    return predictions


def run_policy(dates, returns, forecast, bounds, mode):
    left, right = bounds
    wealth = 1.0
    peak = 1.0
    prior_exposure = 0.0
    rows = []
    for index in range(left, right - 1, 5):
        prediction = forecast[index]
        if mode == "forecast_binary":
            exposure = 1.0 if prediction > 0 else 0.0
        elif mode == "forecast_sized":
            exposure = float(np.clip(0.5 + prediction / 0.02 * 0.5, 0.0, 1.0))
        elif mode == "buy_hold":
            exposure = 1.0
        elif mode == "cash":
            exposure = 0.0
        else:
            raise ValueError(mode)
        next_index = min(index + 5, right - 1)
        period_days = next_index - index
        gross_return = float(np.prod(1.0 + exposure * returns[index + 1 : next_index + 1]) - 1.0)
        turnover = abs(exposure - prior_exposure)
        growth = (1.0 + gross_return) * max(1e-8, 1.0 - 0.0005 * turnover)
        wealth *= growth
        peak = max(peak, wealth)
        rows.append({
            "date": str(dates[index]),
            "prediction_5d": float(prediction),
            "realized_5d_return": gross_return,
            "exposure": exposure,
            "turnover": turnover,
            "wealth": wealth,
            "drawdown": wealth / peak - 1.0,
            "period_days": period_days,
        })
        prior_exposure = exposure
    return rows


def summarize(rows):
    returns = np.asarray([row["wealth"] for row in rows], dtype=np.float64)
    period_returns = np.asarray([
        (1.0 + row["realized_5d_return"]) * (1.0 - 0.0005 * row["turnover"]) - 1.0
        for row in rows
    ])
    period_days = np.asarray([row["period_days"] for row in rows], dtype=np.float64)
    years = period_days.sum() / 252.0
    period_std = float(np.std(period_returns, ddof=1))
    annualizer = np.sqrt(252.0 / float(period_days.mean()))
    return {
        "total_return_pct": float((returns[-1] - 1.0) * 100),
        "cagr_pct": float((returns[-1] ** (1.0 / years) - 1.0) * 100),
        "annualized_volatility_pct": float(period_std * annualizer * 100),
        "sharpe_zero_rate": float(np.mean(period_returns) / period_std * annualizer) if period_std > 0 else 0.0,
        "max_drawdown_pct": float(min(row["drawdown"] for row in rows) * 100),
        "mean_exposure_pct": float(np.mean([row["exposure"] for row in rows]) * 100),
        "annual_turnover": float(sum(row["turnover"] for row in rows) / years),
    }


def main():
    parser = argparse.ArgumentParser(description="Expanding-window ridge forecast of five-session SPY returns.")
    parser.add_argument("--prices", type=Path, default=PRICE_FILE)
    parser.add_argument("--output", type=Path, default=Path("outputs/walkforward-forecast-001"))
    args = parser.parse_args()
    dates, close, daily_returns, base_features = load_series(args.prices)
    ensemble_factors, _ = stockalgo_features(close, daily_returns)
    features = np.column_stack([base_features, ensemble_factors[:, 9:]]).astype(np.float64)
    valid = np.isfinite(features).all(axis=1)
    target = np.full(len(dates), np.nan, dtype=np.float64)
    for index in range(len(dates) - 5):
        target[index] = np.prod(1.0 + daily_returns[index + 1 : index + 6]) - 1.0

    validation = dated_bounds(dates, "2013-01-01", "2018-12-31")
    test = dated_bounds(dates, "2019-01-01", str(dates[-1]))
    first_train = int(np.searchsorted(dates, np.datetime64("1996-01-01"), side="left"))
    predictions = forecasts_walk_forward(
        features, target, valid, validation[0], test[1], first_train=first_train
    )
    predictions[:validation[0]] = np.nan

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    all_rows = []
    for split, bounds in (("validation", validation), ("test", test)):
        split_summary = {}
        for policy in ("forecast_binary", "forecast_sized", "buy_hold", "cash"):
            rows = run_policy(dates, daily_returns, predictions, bounds, policy)
            for row in rows:
                row["split"] = split
                row["policy"] = policy
            all_rows.extend(rows)
            split_summary[policy] = summarize(rows)
        summary_rows = [row for row in all_rows if row["split"] == split]
        y = np.asarray([row["realized_5d_return"] for row in summary_rows if row["policy"] == "buy_hold"])
        p = predictions[bounds[0] : bounds[1] - 1 : 5]
        p = p[: len(y)]
        split_summary["forecast_quality"] = {
            "observations": int(len(p)),
            "direction_accuracy_pct": float(np.mean((p > 0) == (y > 0)) * 100),
            "always_positive_direction_accuracy_pct": float(np.mean(y > 0) * 100),
            "pearson_correlation": float(np.corrcoef(p, y)[0, 1]),
            "mean_absolute_error_pct": float(np.mean(np.abs(p - y)) * 100),
            "mean_forecast_pct": float(np.mean(p) * 100),
            "mean_realized_return_pct": float(np.mean(y) * 100),
        }
        summaries[split] = split_summary

    with (args.output / "weekly_forecasts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "policy", "date", "prediction_5d", "realized_5d_return", "exposure", "turnover", "wealth", "drawdown", "period_days"])
        writer.writeheader()
        writer.writerows(all_rows)
    test_rows = [row for row in all_rows if row["split"] == "test"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for policy in ("forecast_sized", "buy_hold"):
        rows = [row for row in test_rows if row["policy"] == policy]
        axes[0].plot([np.datetime64(row["date"]) for row in rows], [row["wealth"] for row in rows], label=policy.replace("_", " "))
    axes[0].set_title("Walk-forward five-session forecast sizing — untouched test")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    rows = [row for row in test_rows if row["policy"] == "forecast_sized"]
    axes[1].step([np.datetime64(row["date"]) for row in rows], [row["exposure"] for row in rows], where="post", color="#4c78a8")
    axes[1].set_title("Forecast-based target exposure")
    axes[1].set_ylabel("Exposure")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0, 0.5, 1])
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    figure.savefig(args.output / "forecast_test.png", dpi=170, facecolor="white")
    plt.close(figure)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "run.json").write_text(json.dumps({
        "model": "expanding-window ridge regression, refit every 21 sessions",
        "target": "SPY next-five-session adjusted return",
        "ridge_alpha": 25.0,
        "exposure_mapping": "clip(0.5 + forecast / 0.02 * 0.5, 0, 1)",
        "features": list(FEATURE_NAMES),
        "training_window_sessions": 2520,
        "minimum_training_examples": 500,
        "training_start": str(dates[first_train]),
        "label_availability": "A training row is used only after its full five-session return is observed.",
        "validation_period": [str(dates[validation[0]]), str(dates[validation[1] - 1])],
        "test_period": [str(dates[test[0]]), str(dates[test[1] - 1])],
        "transaction_cost_bps_per_unit_turnover": 5,
        "price_source": str(args.prices.resolve().relative_to(ROOT)) if args.prices.resolve().is_relative_to(ROOT) else args.prices.name,
        "limitations": ["uses reconstructed SPY technical ensemble features, not StockAlgo email picks", "one chronological holdout and a fixed ridge setup", "cash earns zero"]
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Results: {args.output / 'weekly_forecasts.csv'}")


if __name__ == "__main__":
    main()
