"""Download candidate historical price windows from Tiingo's EOD API."""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .recovery import ROOT


API = "https://api.tiingo.com/tiingo/daily"
OUTPUT = ROOT / "data/tiingo"
JOURNAL = OUTPUT / "attempts.jsonl"
FIELDS = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]


def missing_windows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        if row.get("price_status") == "price_window_complete":
            continue
        symbol = row["ticker"].strip()
        if not symbol:
            continue
        start = date.fromisoformat(row["price_from"])
        end = date.fromisoformat(row["price_through"])
        current = grouped.setdefault(symbol, {"start": start, "end": end})
        current["start"] = min(current["start"], start)
        current["end"] = max(current["end"], end)
    return grouped


def membership_starts(path):
    starts = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            for symbol in row["tickers"].split(","):
                symbol = symbol.strip()
                if symbol:
                    starts[symbol] = min(day, starts.get(symbol, day))
    return starts


def wait_for_slot(records, delay):
    now = datetime.now(timezone.utc).timestamp()
    timestamps = []
    for record in records:
        try:
            value = datetime.fromisoformat(record["retrieved_at"]).timestamp()
        except (KeyError, TypeError, ValueError):
            continue
        if value > now - 3600:
            timestamps.append(value)
    timestamps.sort()
    waits = [0.0]
    if timestamps:
        waits.append(timestamps[-1] + max(73.0, delay) - now)
    if len(timestamps) >= 50:
        waits.append(timestamps[-50] + 3601 - now)
    wait = max(waits)
    if wait > 0:
        time.sleep(wait)


def benchmark_sessions():
    path = ROOT / "data/raw/SPY_yahoo_daily.csv"
    if not path.exists():
        path = ROOT / "data/raw/spy_yahoo_daily.csv"
    with path.open(newline="") as stream:
        return sorted({
            date.fromisoformat(row["date"])
            for row in csv.DictReader(stream)
            if row.get("date")
        })


def request_json(url, token):
    request = Request(url, headers={
        "Authorization": "Token " + token,
        "Accept": "application/json",
        "User-Agent": "RLPMResearch/1.0",
    })
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def parse_prices(payload):
    rows = {}
    for item in payload:
        try:
            day = date.fromisoformat(item["date"][:10]).isoformat()
            values = {
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": float(item["volume"]),
                "adjusted_close": float(item["adjClose"]),
            }
            if not all(math.isfinite(value) for value in values.values()):
                continue
            if values["adjusted_close"] <= 0:
                continue
            rows[day] = {"date": day, **values}
        except (KeyError, TypeError, ValueError):
            continue
    return [rows[key] for key in sorted(rows)]


def read_journal():
    records = []
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_candidate(symbol, rows, full_history=False):
    suffix = "__full" if full_history else ""
    destination = OUTPUT / (quote(symbol, safe="") + suffix + ".csv")
    temporary = destination.with_suffix(".csv.part")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def run(limit, delay, full_history=False):
    token = os.environ.get("TIINGO_API_KEY", "").strip()
    if not token:
        raise SystemExit(
            "TIINGO_API_KEY is not set. Set it privately in this terminal, then rerun."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    windows = missing_windows(ROOT / "outputs/removal-data-needed.csv")
    sessions = benchmark_sessions()
    records = read_journal()
    scope = "membership_history_v1" if full_history else "audit_window"
    if full_history:
        starts = membership_starts(ROOT / "data/raw/membership.csv")
        for symbol, window in windows.items():
            window["start"] = min(window["start"], starts.get(symbol, window["start"]))
    latest = {
        record.get("ticker"): record for record in records
        if record.get("scope", "audit_window") == scope
    }
    final_statuses = {
        "candidate_downloaded", "partial_candidate", "empty_response", "unsupported_symbol"
    }
    jobs = [symbol for symbol in sorted(windows)
            if latest.get(symbol, {}).get("status") not in final_statuses]
    month = datetime.now().strftime("%Y-%m")
    this_month = {
        record.get("ticker") for record in records
        if str(record.get("retrieved_at", "")).startswith(month)
    }
    new_symbols_left = max(0, limit - len(this_month))
    already_requested = set(latest)
    retry_symbols = [symbol for symbol in jobs if symbol in already_requested]
    previously_used = [symbol for symbol in jobs
                       if symbol not in already_requested and symbol in this_month]
    new_symbols = [symbol for symbol in jobs
                   if symbol not in already_requested and symbol not in this_month]
    jobs = retry_symbols + previously_used + new_symbols[:new_symbols_left]
    if full_history:
        useful = {record.get("ticker") for record in records if record.get("file")}
        jobs.sort(key=lambda symbol: (symbol not in useful, symbol))
    print(f"{len(jobs)} tickers queued (monthly unique-symbol cap: {limit}).", flush=True)
    counts = Counter()

    for index, symbol in enumerate(jobs, 1):
        wait_for_slot(read_journal(), delay)
        start, end = windows[symbol]["start"], windows[symbol]["end"]
        params = {"startDate": start.isoformat()}
        if not full_history:
            params["endDate"] = end.isoformat()
        url = f"{API}/{quote(symbol, safe='')}/prices?" + urlencode(params)
        record = {
            "ticker": symbol,
            "requested_from": start.isoformat(),
            "requested_through": end.isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "usage_month": month,
            "scope": scope,
            "identity_status": "unchecked",
            "settlement_status": "unchecked",
        }
        stop = False
        while True:
            try:
                payload = request_json(url, token)
                break
            except HTTPError as error:
                if error.code == 429:
                    raw = error.headers.get("Retry-After", "")
                    try:
                        wait = max(60, int(raw))
                    except ValueError:
                        wait = 3600
                    print(f"Tiingo rate limit; waiting {wait} seconds, then retrying {symbol}.", flush=True)
                    time.sleep(wait)
                    continue
                record.update(status="unsupported_symbol" if error.code == 404 else "http_error",
                              http_code=error.code)
                stop = error.code in (401, 403)
                payload = None
                break
            except (URLError, TimeoutError, OSError, ValueError, TypeError) as error:
                record.update(status="request_failed", error=type(error).__name__)
                payload = None
                break

        try:
            if payload is not None:
                rows = parse_prices(payload)
                if not rows:
                    record["status"] = "empty_response"
                else:
                    returned = {date.fromisoformat(row["date"]) for row in rows}
                    expected_end = date.fromisoformat(rows[-1]["date"]) if full_history else end
                    expected = {
                        session for session in sessions
                        if start <= session <= expected_end
                    }
                    covered = returned & expected
                    path = write_candidate(symbol, rows, full_history)
                    record.update(
                        status="candidate_downloaded" if expected <= returned else "partial_candidate",
                        rows=len(rows),
                        first_date=rows[0]["date"],
                        last_date=rows[-1]["date"],
                        expected_sessions=len(expected),
                        sessions_covered=len(covered),
                        file=str(path.relative_to(ROOT)),
                        source="Tiingo EOD API",
                    )
        except (URLError, TimeoutError, OSError, ValueError, TypeError) as error:
            record.update(status="request_failed", error=type(error).__name__)

        with JOURNAL.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        counts[record["status"]] += 1
        if index % 10 == 0 or index == len(jobs) or stop:
            summary = " | ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
            print(f"Processed {index}/{len(jobs)} | {summary}", flush=True)
        if stop:
            print("Stopped safely; saved progress can be resumed later.", flush=True)
            break
        if index < len(jobs):
            time.sleep(delay)

    print(f"Candidate prices: {OUTPUT}")
    print(f"Request journal: {JOURNAL}")
    print("Downloaded rows still need ticker identity and corporate-action review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=500,
        help="maximum unique symbols this run may request; Tiingo Starter permits 500/month",
    )
    parser.add_argument("--delay", type=float, default=73.0,
                        help="seconds between requests (default keeps below 50 requests/hour)")
    parser.add_argument("--full-history", action="store_true",
                        help="download from first recorded membership through the last available price")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 500:
        parser.error("--limit must be between 1 and 500 on the free Starter plan")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be zero or greater")
    run(args.limit, args.delay, args.full_history)


if __name__ == "__main__":
    main()
