from bisect import bisect_right
from collections import Counter
import csv
from datetime import date, datetime, timezone
import json

from .data_readiness import ROOT, RAW, COMBINED, membership_price_dates
from .prices import valid_dates


def run():
    output = ROOT / "outputs"
    benchmark = next(path for path in (
        RAW / "SPY_yahoo_daily.csv", RAW / "spy_yahoo_daily.csv"
    ) if path.exists())
    sessions = sorted(valid_dates(benchmark)[0])
    with (RAW / "membership.csv").open(newline="") as stream:
        snapshots = sorted((date.fromisoformat(row["date"]), set(row["tickers"].split(",")))
                           for row in csv.DictReader(stream))
    starts = [day for day, members in snapshots]
    sessions = [day for day in sessions if day >= starts[0]]
    rosters = [snapshots[bisect_right(starts, day) - 1][1] for day in sessions]
    with (output / "data-readiness.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    rules_path = ROOT / "data/membership-price-identities.csv"
    with rules_path.open(newline="") as stream:
        rules = list(csv.DictReader(stream))
    windows = []
    by_year = Counter()
    for row in summary:
        if not int(row["price_sessions_missing"]):
            continue
        ticker = row["ticker"]
        observed, invalid, duplicates = membership_price_dates(ticker, rules)
        first = min(observed) if observed else None
        last = max(observed) if observed else None
        active = None
        for day, members in zip(sessions, rosters):
            missing = ticker in members and day not in observed
            reason = None
            if missing:
                by_year[day.year] += 1
                reason = ("no_saved_prices" if not observed else "before_saved_history"
                          if day < first else "after_saved_history" if day > last
                          else "internal_gap")
            if active and (not missing or reason != active["gap_type"]):
                windows.append(active)
                active = None
            if missing:
                if active is None:
                    active = {"ticker": ticker, "from": day.isoformat(),
                              "through": day.isoformat(), "missing_sessions": 0,
                              "gap_type": reason}
                active["through"] = day.isoformat()
                active["missing_sessions"] += 1
        if active:
            windows.append(active)
    windows.sort(key=lambda row: (-row["missing_sessions"], row["ticker"], row["from"]))
    with (output / "missing-price-windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "ticker", "from", "through", "missing_sessions", "gap_type"
        ])
        writer.writeheader()
        writer.writerows(windows)
    expected = sum(int(row["membership_sessions"]) for row in summary)
    found = sum(int(row["price_sessions_found"]) for row in summary)
    if sum(row["missing_sessions"] for row in windows) != expected - found:
        raise ValueError("Gap report does not match the readiness audit")
    with (COMBINED / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    with (COMBINED / "source-alignment.csv").open(newline="") as stream:
        alignments = list(csv.DictReader(stream))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tickers_with_prices": len(manifest),
        "daily_rows": sum(int(row["rows"]) for row in manifest),
        "membership_price_pairs": expected,
        "covered_pairs": found,
        "coverage_pct": 100 * found / expected,
        "missing_pairs": expected - found,
        "remaining_gap_windows": len(windows),
        "missing_sessions_by_year": dict(sorted(by_year.items())),
        "source_alignment": dict(Counter(row["status"] for row in alignments)),
        "identifiers_verified": False,
        "terminal_returns_verified": False,
        "training_ready": False,
    }
    (output / "data-status.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
