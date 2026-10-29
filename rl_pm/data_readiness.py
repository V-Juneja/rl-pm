import csv
import math
from bisect import bisect_right
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COMBINED = RAW / "combined"
OUTPUT = ROOT / "outputs" / "data-readiness.csv"


def membership_price_dates(ticker, rules):
    path = COMBINED / f"{ticker}_combined_daily.csv"
    dates, invalid, duplicates = read_price_dates(path) if path.exists() else (set(), 0, 0)
    for rule in rules:
        if rule["membership_ticker"] != ticker:
            continue
        first = date.fromisoformat(rule["from"])
        last = date.fromisoformat(rule["through"])
        source = (ROOT / rule["price_file"] if rule.get("price_file")
                  else COMBINED / f"{rule['price_ticker']}_combined_daily.csv")
        dates = {day for day in dates if not first <= day <= last}
        if source.exists():
            available, bad, repeated = read_price_dates(source)
            dates.update(day for day in available if first <= day <= last)
            invalid += bad
            duplicates += repeated
    return dates, invalid, duplicates


def read_price_dates(path):
    dates = set()
    invalid_rows = 0
    duplicate_dates = 0

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                day = date.fromisoformat(row["date"])
                price = float(row["adjusted_close"])
                if not math.isfinite(price) or price <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
                continue

            if day in dates:
                duplicate_dates += 1
            dates.add(day)

    return dates, invalid_rows, duplicate_dates


def main():
    membership_path = RAW / "membership.csv"
    if not membership_path.exists():
        raise SystemExit(f"Missing file: {membership_path}")

    snapshots = []
    with membership_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            tickers = {ticker.strip() for ticker in row["tickers"].split(",") if ticker.strip()}
            snapshots.append((day, tickers))

    snapshots.sort(key=lambda item: item[0])
    snapshot_dates = [item[0] for item in snapshots]

    spy_candidates = [
        RAW / "SPY_yahoo_daily.csv",
        RAW / "spy_yahoo_daily.csv",
        COMBINED / "SPY_combined_daily.csv",
        COMBINED / "spy_combined_daily.csv",
    ]
    spy_path = next((path for path in spy_candidates if path.exists()), None)
    if spy_path is None:
        raise SystemExit("Could not find an SPY price CSV in data/raw or data/raw/combined")

    sessions, spy_invalid, spy_duplicates = read_price_dates(spy_path)
    sessions = sorted(day for day in sessions if day >= snapshot_dates[0])
    if not sessions:
        raise SystemExit("No SPY sessions overlap the membership history")

    price_cache = {}
    invalid_by_ticker = Counter()
    duplicates_by_ticker = Counter()
    missing_by_ticker = Counter()
    expected_by_ticker = Counter()
    first_missing = {}
    rules_path = ROOT / "data/membership-price-identities.csv"
    if rules_path.exists():
        with rules_path.open(newline="") as stream:
            rules = list(csv.DictReader(stream))
    else:
        rules = []

    for session in sessions:
        snapshot_index = bisect_right(snapshot_dates, session) - 1
        if snapshot_index < 0:
            continue

        members = snapshots[snapshot_index][1]
        for ticker in members:
            expected_by_ticker[ticker] += 1

            if ticker not in price_cache:
                dates, invalid, duplicates = membership_price_dates(ticker, rules)
                price_cache[ticker] = dates
                invalid_by_ticker[ticker] = invalid
                duplicates_by_ticker[ticker] = duplicates

            if session not in price_cache[ticker]:
                missing_by_ticker[ticker] += 1
                first_missing.setdefault(ticker, session)

    manifest_path = COMBINED / "manifest.csv"
    identity_statuses = Counter()
    settlement_statuses = Counter()

    if manifest_path.exists():
        with manifest_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                identity_statuses[row.get("identity_status", "missing")] += 1
                settlement_statuses[row.get("settlement_status", "missing")] += 1

    rows = []
    for ticker in sorted(expected_by_ticker):
        expected = expected_by_ticker[ticker]
        missing = missing_by_ticker[ticker]
        rows.append({
            "ticker": ticker,
            "membership_sessions": expected,
            "price_sessions_found": expected - missing,
            "price_sessions_missing": missing,
            "coverage_pct": round(100 * (expected - missing) / expected, 2),
            "first_missing_date": first_missing.get(ticker, ""),
            "invalid_price_rows": invalid_by_ticker[ticker],
            "duplicate_price_dates": duplicates_by_ticker[ticker],
            "combined_price_file_found": bool(price_cache[ticker]),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_expected = sum(expected_by_ticker.values())
    total_missing = sum(missing_by_ticker.values())
    covered = total_expected - total_missing

    print(f"Membership dates: {len(snapshots):,}")
    print(f"SPY sessions checked: {len(sessions):,}")
    print(f"Tickers checked: {len(expected_by_ticker):,}")
    print(f"Membership-day prices found: {covered:,} / {total_expected:,} ({100 * covered / total_expected:.2f}%)")
    print(f"Membership-day prices missing: {total_missing:,}")
    print(f"Invalid SPY rows: {spy_invalid:,}")
    print(f"Duplicate SPY dates: {spy_duplicates:,}")
    print(f"Identity status counts: {dict(identity_statuses)}")
    print(f"Delisting-settlement status counts: {dict(settlement_statuses)}")
    print(f"Report: {OUTPUT}")
    print("This checks price-date coverage only; it does not verify ticker identity, delisting returns, or training readiness.")


if __name__ == "__main__":
    main()
