import argparse
from bisect import bisect_left
from collections import Counter
import csv
from datetime import date
import math
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUTPUT = ROOT / "outputs"


def read_prices(path):
    prices = {}

    with path.open() as file:
        reader = csv.DictReader(file)

        if not {"date", "adjusted_close"} <= set(
            reader.fieldnames or []
        ):
            raise ValueError(f"Invalid price columns: {path.name}")

        for row in reader:
            try:
                day = date.fromisoformat(row["date"])
                value = float(row["adjusted_close"])
            except (ValueError, TypeError):
                continue

            if math.isfinite(value) and value > 0:
                prices[day] = value

    return prices


def read_removals(path):
    with path.open() as file:
        snapshots = [
            (
                date.fromisoformat(row["date"]),
                {
                    symbol.strip()
                    for symbol in row["tickers"].split(",")
                    if symbol.strip()
                },
            )
            for row in csv.DictReader(file)
        ]

    if not snapshots:
        raise ValueError("Membership file is empty")

    snapshots.sort(key=lambda row: row[0])

    dates = [day for day, _ in snapshots]

    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate membership dates")

    removals = []

    for previous, current in zip(snapshots, snapshots[1:]):
        previous_members = previous[1]
        day, current_members = current

        for symbol in sorted(previous_members - current_members):
            removals.append((day, symbol))

    return removals


def price_path(symbol):
    return DATA / (
        quote(symbol, safe="") + "_yahoo_daily.csv"
    )


def benchmark_path():
    for filename in (
        "SPY_yahoo_daily.csv",
        "spy_yahoo_daily.csv",
    ):
        path = DATA / filename

        if path.exists():
            return path

    raise FileNotFoundError("SPY price file not found")


def inspect_removal(day, symbol, calendar, horizon, cache):
    position = bisect_left(calendar, day)
    before = calendar[position - 1] if position else None
    after = calendar[position:position + horizon]
    path = price_path(symbol)

    result = {
        "ticker": symbol,
        "removal_snapshot": day.isoformat(),
        "status": "",
        "pre_removal_session": before.isoformat() if before else "",
        "pre_removal_price_present": False,
        "post_sessions_checked": len(after),
        "post_sessions_missing": len(after),
        "first_missing_session": after[0].isoformat() if after else "",
        "last_available_price": "",
        "return_over_checked_window": "",
        "identifier_status": "unchecked",
        "settlement_status": "unchecked",
        "error": "",
    }

    if not path.exists():
        result["status"] = "missing_file"
        return result

    if symbol not in cache:
        try:
            cache[symbol] = (read_prices(path), "")
        except (OSError, ValueError, UnicodeError, csv.Error) as error:
            cache[symbol] = ({}, str(error))

    prices, error = cache[symbol]

    if error:
        result["status"] = "unreadable_file"
        result["error"] = error
        return result

    if not prices:
        result["status"] = "no_valid_prices"
        return result

    result["last_available_price"] = max(prices).isoformat()
    result["pre_removal_price_present"] = before in prices

    missing = [session for session in after if session not in prices]
    result["post_sessions_missing"] = len(missing)
    result["first_missing_session"] = (
        missing[0].isoformat() if missing else ""
    )

    if before is None or not after:
        result["status"] = "outside_calendar"
    elif before not in prices:
        result["status"] = "missing_pre_removal_price"
    elif len(after) < horizon:
        result["status"] = "short_benchmark_history"
    elif missing:
        result["status"] = "post_removal_gaps"
    else:
        result["status"] = "price_window_complete"
        result["return_over_checked_window"] = round(
            prices[after[-1]] / prices[before] - 1,
            8,
        )

    return result


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(1998, 1, 1),
    )
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--sessions", type=int, default=20)
    args = parser.parse_args()

    if args.sessions < 1:
        parser.error("--sessions must be at least 1")

    calendar = sorted(read_prices(benchmark_path()))

    if not calendar:
        parser.error("SPY price file has no valid observations")

    end = args.end or calendar[-1]

    if end < args.start:
        parser.error("--end must not precede --start")

    removals = [
        (day, symbol)
        for day, symbol in read_removals(DATA / "membership.csv")
        if args.start <= day <= end
    ]

    print(f"Checking {len(removals):,} removal events...", flush=True)

    cache = {}
    results = [
        inspect_removal(
            day, symbol, calendar, args.sessions, cache
        )
        for day, symbol in removals
    ]

    if not results:
        print("No removal events in this period.")
        return

    OUTPUT.mkdir(exist_ok=True)
    report = OUTPUT / "removal-audit.csv"
    save_csv(report, results)

    needed = []

    for row in results:
        day = date.fromisoformat(row["removal_snapshot"])
        position = bisect_left(calendar, day)
        final_position = min(
            position + args.sessions - 1,
            len(calendar) - 1,
        )

        needed.append({
            "ticker": row["ticker"],
            "removal_snapshot": row["removal_snapshot"],
            "price_status": row["status"],
            "price_from": (
                row["pre_removal_session"]
                or row["removal_snapshot"]
            ),
            "price_through": calendar[final_position].isoformat(),
            "price_history_needed": (
                row["status"] != "price_window_complete"
            ),
            "identifier_status": "unchecked",
            "settlement_status": "unchecked",
        })

    requirements = OUTPUT / "removal-data-needed.csv"
    save_csv(requirements, needed)

    counts = Counter(row["status"] for row in results)

    print("\nrl-pm removal audit\n")
    print(f"{'Removal events':<32} {len(results):>7,}")
    print(
        f"{'Distinct removed tickers':<32} "
        f"{len({row['ticker'] for row in results}):>7,}"
    )

    for status, count in sorted(counts.items()):
        print(f"{status:<32} {count:>7,}")

    print(f"\nResults: {report}")
    print(f"Data requirements: {requirements}")


if __name__ == "__main__":
    main()
