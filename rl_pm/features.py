import argparse
from collections import deque
import csv
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COMBINED = RAW / "combined"

BASE_NAMES = (
    "log_return_1",
    "log_return_5",
    "log_return_21",
    "log_return_63",
    "log_return_252",
    "relative_log_return_21",
    "relative_log_return_63",
    "volatility_21",
    "volatility_63",
    "downside_volatility_21",
    "distance_from_mean_21",
    "distance_from_mean_63",
    "distance_from_mean_252",
    "drawdown_252",
)

FEATURE_NAMES = (
    *BASE_NAMES,
    *(f"{name}_available" for name in BASE_NAMES),
    "price_available",
    "continuous_history_252",
)


def clean_prices(values):
    values = np.asarray(values, dtype=np.float64).copy()
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def rolling_sum(values, window):
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("rolling_sum expects a one-dimensional array")

    if window < 1:
        raise ValueError("Window must be positive")

    result = np.full(len(values), np.nan, dtype=np.float64)

    if len(values) < window:
        return result

    totals = np.empty(len(values) + 1, dtype=np.float64)
    totals[0] = 0.0
    np.cumsum(values, out=totals[1:])

    result[window - 1:] = totals[window:] - totals[:-window]

    return result


def rolling_mean(values, window):
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)

    totals = rolling_sum(safe, window)
    counts = rolling_sum(valid.astype(np.float64), window)

    result = totals / window
    result[counts != window] = np.nan

    return result


def rolling_std(values, window):
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)

    totals = rolling_sum(safe, window)
    squares = rolling_sum(safe * safe, window)
    counts = rolling_sum(valid.astype(np.float64), window)

    mean = totals / window
    variance = np.maximum(squares / window - mean * mean, 0.0)

    result = np.sqrt(variance)
    result[counts != window] = np.nan

    return result


def rolling_max(values, window):
    result = np.full(len(values), np.nan, dtype=np.float64)
    candidates = deque()

    valid = np.isfinite(values)
    counts = rolling_sum(valid.astype(np.float64), window)

    for index, value in enumerate(values):
        first = index - window + 1

        while candidates and candidates[0] < first:
            candidates.popleft()

        if valid[index]:
            while candidates and values[candidates[-1]] <= value:
                candidates.pop()

            candidates.append(index)

        if index >= window - 1 and counts[index] == window:
            result[index] = values[candidates[0]]

    return result


def trailing_log_return(prices, window):
    result = np.full(len(prices), np.nan, dtype=np.float64)

    if len(prices) <= window:
        return result

    valid = np.isfinite(prices)
    counts = rolling_sum(valid.astype(np.float64), window + 1)

    positions = np.arange(window, len(prices))
    positions = positions[counts[positions] == window + 1]

    result[positions] = (
        np.log(prices[positions])
        - np.log(prices[positions - window])
    )

    return result


def continuous_history(prices, maximum=252):
    result = np.zeros(len(prices), dtype=np.float64)
    count = 0

    for index, value in enumerate(prices):
        if np.isfinite(value):
            count = min(count + 1, maximum)
        else:
            count = 0

        result[index] = count / maximum

    return result


def stock_features(prices, benchmark):
    prices = clean_prices(prices)
    benchmark = clean_prices(benchmark)

    if prices.ndim != 1 or benchmark.ndim != 1:
        raise ValueError("Prices and benchmark must be one-dimensional")

    if prices.shape != benchmark.shape:
        raise ValueError("Prices and benchmark must use the same sessions")

    daily = trailing_log_return(prices, 1)
    columns = []

    for window in (1, 5, 21, 63, 252):
        columns.append(trailing_log_return(prices, window))

    for window in (21, 63):
        stock_return = trailing_log_return(prices, window)
        market_return = trailing_log_return(benchmark, window)
        columns.append(stock_return - market_return)

    for window in (21, 63):
        columns.append(rolling_std(daily, window) * np.sqrt(252.0))

    downside = np.minimum(daily, 0.0)
    downside_variance = rolling_mean(downside * downside, 21)
    columns.append(np.sqrt(downside_variance) * np.sqrt(252.0))

    for window in (21, 63, 252):
        average = rolling_mean(prices, window)
        columns.append(prices / average - 1.0)

    high = rolling_max(prices, 252)
    columns.append(prices / high - 1.0)

    raw = np.column_stack(columns)
    available = np.isfinite(raw)

    result = np.zeros(
        (len(prices), len(FEATURE_NAMES)),
        dtype=np.float32,
    )

    feature_count = len(BASE_NAMES)

    result[:, :feature_count] = np.where(
        available,
        raw,
        0.0,
    ).astype(np.float32)

    result[:, feature_count:feature_count * 2] = available
    result[:, -2] = np.isfinite(prices)
    result[:, -1] = continuous_history(prices)

    return result


def build_features(adjusted_close, benchmark_adjusted_close):
    prices = np.asarray(adjusted_close)
    benchmark = np.asarray(benchmark_adjusted_close)

    if prices.ndim == 1:
        prices = prices[:, None]

    if prices.ndim != 2:
        raise ValueError("adjusted_close must have shape T,N")

    if min(prices.shape) < 1:
        raise ValueError("Price matrix cannot be empty")

    if benchmark.shape != (prices.shape[0],):
        raise ValueError("Benchmark must have shape T")

    result = np.empty(
        (
            prices.shape[0],
            prices.shape[1],
            len(FEATURE_NAMES),
        ),
        dtype=np.float32,
    )

    for column in range(prices.shape[1]):
        result[:, column, :] = stock_features(
            prices[:, column],
            benchmark,
        )

    return result


def read_prices(path):
    prices = {}

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                day = date.fromisoformat(row["date"][:10])
                value = float(row["adjusted_close"])
            except (KeyError, TypeError, ValueError):
                continue

            if not np.isfinite(value) or value <= 0:
                continue

            if day in prices and prices[day] != value:
                raise ValueError(
                    f"Conflicting prices for {day} in {path.name}"
                )

            prices[day] = value

    return prices


def benchmark_path():
    candidates = (
        RAW / "SPY_yahoo_daily.csv",
        RAW / "spy_yahoo_daily.csv",
        COMBINED / "SPY_combined_daily.csv",
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("SPY prices were not found")


def load_inputs(tickers, start, end):
    benchmark = read_prices(benchmark_path())

    sessions = sorted(
        day
        for day in benchmark
        if start <= day and (end is None or day <= end)
    )

    if not sessions:
        raise ValueError("No benchmark sessions in the selected range")

    index = {day: position for position, day in enumerate(sessions)}

    matrix = np.full(
        (len(sessions), len(tickers)),
        np.nan,
        dtype=np.float64,
    )

    for column, ticker in enumerate(tickers):
        filename = quote(ticker, safe="") + "_combined_daily.csv"
        path = COMBINED / filename

        if not path.exists():
            raise FileNotFoundError(f"No combined prices for {ticker}")

        for day, value in read_prices(path).items():
            position = index.get(day)

            if position is not None:
                matrix[position, column] = value

    benchmark_values = np.asarray(
        [benchmark[day] for day in sessions],
        dtype=np.float64,
    )

    return (
        np.asarray(sessions, dtype="datetime64[D]"),
        matrix,
        benchmark_values,
    )


def show_latest(features, dates, tickers):
    feature_count = len(BASE_NAMES)

    print(f"Latest session: {dates[-1]}")
    print(f"Feature tensor: {features.shape}")
    print()

    for column, ticker in enumerate(tickers):
        print(ticker)

        for index, name in enumerate(BASE_NAMES):
            available = features[-1, column, feature_count + index]

            if available:
                value = f"{features[-1, column, index]:.6f}"
            else:
                value = "missing"

            print(f"  {name:<30} {value:>12}")

        print(
            f"  {'price_available':<30} "
            f"{features[-1, column, -2]:>12.0f}"
        )
        print(
            f"  {'continuous_history_252':<30} "
            f"{features[-1, column, -1]:>12.6f}"
        )
        print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(1996, 1, 2),
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "features-preview.npz",
    )

    args = parser.parse_args()

    if args.end is not None and args.start > args.end:
        parser.error("--start must be on or before --end")

    tickers = list(dict.fromkeys(
        ticker.strip().upper()
        for ticker in args.tickers
        if ticker.strip()
    ))

    if not tickers:
        parser.error("At least one ticker is required")

    dates, prices, benchmark = load_inputs(
        tickers,
        args.start,
        args.end,
    )

    features = build_features(prices, benchmark)

    output = args.output

    if not output.is_absolute():
        output = ROOT / output

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as stream:
        np.savez_compressed(
            stream,
            dates=dates,
            tickers=np.asarray(tickers, dtype=str),
            features=features,
            feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        )

    show_latest(features, dates, tickers)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()