import json
from pathlib import Path
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_pm.features import BASE_NAMES, stock_features


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    axis.set_axisbelow(True)


def main():
    folder = ROOT / "data/raw/final"
    output = ROOT / "outputs/journal"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((folder / "manifest.json").read_text())
    connection = duckdb.connect()
    connection.read_parquet(str(folder / "availability.parquet")).create_view("availability")
    connection.read_parquet(str(folder / "prices.parquet")).create_view("prices")
    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    yearly = connection.execute("""
        SELECT year(date),count(*),sum(CAST(price_available AS INTEGER))
        FROM availability GROUP BY 1 ORDER BY 1
    """).fetchall()
    years = np.asarray([row[0] for row in yearly])
    unavailable = np.asarray([row[1] - row[2] for row in yearly])
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained",
                                gridspec_kw={"width_ratios": [1.15, 1]})
    figure.suptitle("RL-PM | Historical data snapshot", fontsize=19, fontweight="bold")
    axes[0].bar(years, unavailable, color="tab:blue", width=0.75)
    axes[0].set_title("Membership-days with no price observation")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Flagged observations")
    axes[0].set_xticks(years[::5])
    style(axes[0])
    axes[1].axis("off")
    metrics = [
        (f"{metadata['historical_securities']:,}", "historical securities with a price history"),
        (f"{metadata['daily_price_observations']:,}", "daily price observations"),
        (f"{metadata['price_availability_pct']:.4f}%", "membership-day price availability"),
        ("0", "carried-price rows included"),
    ]
    for index, (number, label) in enumerate(metrics):
        top = 0.91 - index * 0.205
        axes[1].text(0.05, top, number, fontsize=25, fontweight="bold",
                     transform=axes[1].transAxes)
        axes[1].text(0.05, top - 0.065, label, fontsize=11,
                     transform=axes[1].transAxes)
    counts = metadata["availability_counts"]
    axes[1].text(0.05, 0.025,
                 f"{counts['source_reported_no_quote']} source-reported no-quote dates\n"
                 f"{counts['after_last_available_price']} dates after the last available price",
                 transform=axes[1].transAxes, fontsize=10)
    figure.supxlabel(
        f"{metadata['membership_start']} to {metadata['membership_end']} | "
        "Dated security identifiers | Identity and terminal-payout audits pending",
        fontsize=10,
    )
    figure.savefig(output / "01-data-coverage.png", dpi=160)
    plt.close(figure)
    rows = connection.execute("""
        SELECT p.date,p.adjusted_close,b.adjusted_close
        FROM prices p JOIN prices b ON p.date=b.date AND b.security_id='SPY'
        WHERE p.security_id='AAPL' ORDER BY p.date
    """).fetchall()
    dates = np.asarray([row[0] for row in rows])
    prices = np.asarray([row[1] for row in rows], dtype=float)
    benchmark = np.asarray([row[2] for row in rows], dtype=float)
    features = stock_features(prices, benchmark)
    selected = np.asarray([day.year >= 2023 for day in dates])
    timeline = dates[selected]
    selected_features = features[selected]
    feature_indices = {name: index for index, name in enumerate(BASE_NAMES)}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained", sharex=True)
    figure.suptitle("RL-PM | Backward-looking model inputs", fontsize=19, fontweight="bold")
    rebased = prices[selected] / prices[selected][0] * 100
    axes[0, 0].plot(timeline, rebased, color="black", linewidth=1.1)
    axes[0, 0].set_title("AAPL adjusted price")
    axes[0, 0].set_ylabel("Rebased to 100")
    for window, color in [(21, "tab:blue"), (63, "tab:orange")]:
        axes[0, 1].plot(timeline,
                        selected_features[:, feature_indices[f"log_return_{window}"]] * 100,
                        color=color, linewidth=1, label=f"{window} sessions")
    axes[0, 1].axhline(0, color="0.5", linewidth=0.7)
    axes[0, 1].set_title("Trailing momentum")
    axes[0, 1].set_ylabel("Log return (%)")
    axes[0, 1].legend(frameon=False)
    for window, color in [(21, "tab:blue"), (63, "tab:orange")]:
        axes[1, 0].plot(timeline,
                        selected_features[:, feature_indices[f"volatility_{window}"]] * 100,
                        color=color, linewidth=1, label=f"{window} sessions")
    axes[1, 0].set_title("Trailing volatility")
    axes[1, 0].set_ylabel("Annualized log-return volatility (%)")
    axes[1, 0].legend(frameon=False)
    drawdown = selected_features[:, feature_indices["drawdown_252"]] * 100
    axes[1, 1].plot(timeline, drawdown, color="tab:red", linewidth=1)
    axes[1, 1].fill_between(timeline, drawdown, 0, color="tab:red", alpha=0.12)
    axes[1, 1].set_title("Drawdown from the trailing 252-session high")
    axes[1, 1].set_ylabel("Drawdown (%)")
    for axis in axes.flat:
        style(axis)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.supxlabel("AAPL | Saved price observations | Features calculated using current and past observations", fontsize=10)
    figure.savefig(output / "02-training-features.png", dpi=160)
    plt.close(figure)
    for path in sorted(output.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
