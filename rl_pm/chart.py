import argparse
from bisect import bisect_left
import csv
from datetime import date
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .chart_events import EventViewer


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "white"
PANEL = "white"
TEXT = "black"
MUTED = "#444444"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
RED = "#d62728"


def load_prices():
    candidates = [
        ROOT / "data/raw/SPY_yahoo_daily.csv",
        ROOT / "data/raw/spy_yahoo_daily.csv",
    ]
    path = next((path for path in candidates if path.exists()), None)

    if path is None:
        raise ValueError("SPY price file not found")

    prices = {}

    with path.open() as file:
        for row in csv.DictReader(file):
            try:
                day = date.fromisoformat(row["date"])
                price = float(row["adjusted_close"])

                if math.isfinite(price) and price > 0:
                    prices[day] = price
            except (ValueError, TypeError):
                continue

    return sorted(prices.items())


def load_events(tickers):
    path = ROOT / "data/raw/membership.csv"
    snapshots = []

    with path.open() as file:
        for row in csv.DictReader(file):
            members = {
                symbol.strip()
                for symbol in row["tickers"].split(",")
                if symbol.strip()
            }
            snapshots.append((
                date.fromisoformat(row["date"]),
                members,
            ))

    snapshots.sort()

    if not snapshots:
        raise ValueError("Membership file is empty")

    events = []
    previous = snapshots[0][1]

    for day, current in snapshots[1:]:
        added = current - previous
        removed = previous - current

        if tickers is not None:
            added &= tickers
            removed &= tickers

        for ticker in sorted(added):
            events.append((day, ticker, "Added"))

        for ticker in sorted(removed):
            events.append((day, ticker, "Removed"))

        previous = current

    return events, snapshots[-1][0]


def draw_chart(prices, events, start, end, tickers):
    prices = [
        (day, price)
        for day, price in prices
        if start <= day <= end
    ]

    if len(prices) < 2:
        raise ValueError("Not enough prices in this date range")

    dates = [day for day, _ in prices]
    values = [price / prices[0][1] * 100 for _, price in prices]
    events = [
        event for event in events
        if dates[0] <= event[0] <= dates[-1]
    ]

    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.facecolor": BACKGROUND,
    })

    fig, (ax, activity) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [4, 1.5],
            "hspace": 0.12,
        },
    )
    fig.patch.set_facecolor(BACKGROUND)

    for panel in (ax, activity):
        panel.set_facecolor(PANEL)
        panel.grid(axis="y", color="0.88", linewidth=0.5)
        panel.set_axisbelow(True)
        panel.tick_params(length=3, pad=5)

        for spine in panel.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("0.4")

    fig.text(
        0.08, 0.95,
        "rl-pm / S&P 500 membership",
        fontsize=14,
    )
    fig.text(
        0.08, 0.91,
        "SPY adjusted price · Rebased to 100 · "
        "Symbol changes inferred from membership snapshots",
        fontsize=10,
        color=MUTED,
    )

    ax.plot(dates, values, color=BLUE, linewidth=1.6)
    ax.set_ylabel("Growth index · start = 100", labelpad=14)
    ax.margins(x=0.01, y=0.15)
    ax.tick_params(labelbottom=False)

    monthly = {}
    plotted = []

    for index, (day, ticker, action) in enumerate(events):
        position = bisect_left(dates, day)

        if position >= len(dates):
            continue

        market_day = dates[position]
        value = values[position]
        color = GREEN if action == "Added" else RED
        marker = "^" if action == "Added" else "v"

        ax.scatter(
            market_day,
            value,
            color=color,
            marker=marker,
            s=22,
            alpha=0.55,
            linewidths=0,
            zorder=4,
        )

        month = day.replace(day=1)
        counts = monthly.setdefault(
            month, {"Added": 0, "Removed": 0}
        )
        counts[action] += 1

        plotted.append({
            "snapshot_date": day.isoformat(),
            "ticker": ticker,
            "action": action,
            "price_marker_date": market_day.isoformat(),
        })

    months = sorted(monthly)
    added = [monthly[month]["Added"] for month in months]
    removed = [-monthly[month]["Removed"] for month in months]

    activity.bar(
        months, added,
        width=23, color=GREEN, alpha=0.85,
    )
    activity.bar(
        months, removed,
        width=23, color=RED, alpha=0.85,
    )
    activity.axhline(0, color=MUTED, linewidth=0.7)
    activity.set_ylabel("Monthly changes", labelpad=14)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    activity.xaxis.set_major_locator(locator)
    activity.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=BLUE,
                linewidth=2, label="SPY",
            ),
            Line2D(
                [0], [0], marker="^", color=GREEN,
                linestyle="None", label="Symbol added",
            ),
            Line2D(
                [0], [0], marker="v", color=RED,
                linestyle="None", label="Symbol removed",
            ),
        ],
        loc="upper left",
        frameon=False,
        labelcolor=TEXT,
        ncol=3,
    )

    fig.text(
        0.08, 0.035,
        f"{len(plotted):,} symbol events · "
        f"{dates[0]:%b %Y} – {dates[-1]:%b %Y}",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.07, right=0.68, top=0.85, bottom=0.12,
    )

    fig.event_viewer = EventViewer(
        fig, ax, activity, dates, values, events
    )

    return fig, plotted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2000, 1, 1),
    )
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    tickers = (
        list(dict.fromkeys(ticker.upper() for ticker in args.tickers))
        if args.tickers
        else None
    )

    prices = load_prices()
    events, last_snapshot = load_events(
        set(tickers) if tickers else None
    )

    end = args.end or last_snapshot

    if args.start >= end:
        parser.error("--start must be earlier than --end")

    fig, plotted = draw_chart(
        prices, events, args.start, end, tickers
    )

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)

    image = output / "membership-chart.png"
    fig.savefig(image, dpi=200)

    with (output / "membership-events.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=[
            "snapshot_date",
            "ticker",
            "action",
            "price_marker_date",
        ])
        writer.writeheader()
        writer.writerows(plotted)

    print(f"Chart: {image}")
    print(f"Membership events: {len(plotted)}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()