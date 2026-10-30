import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def suspension_intervals(path):
    return [
        {
            "security_id": row["ticker"],
            "last_trade": np.datetime64(row["last_trade"]),
            "next_trade": np.datetime64(row["next_trade"]),
            "source": str(path),
        }
        for row in read_rows(path)
        if row["last_trade"] and row["next_trade"]
    ]


def apply_suspensions(prices, dates, ids, intervals):
    removed = 0
    for interval in intervals:
        index = int(np.searchsorted(ids, interval["security_id"]))
        if index == len(ids) or ids[index] != interval["security_id"]:
            continue
        mask = (dates > interval["last_trade"]) & (dates < interval["next_trade"])
        removed += int(np.isfinite(prices[mask, index]).sum())
        prices[mask, index] = np.nan
    return removed


def cash_entitlement_return(cash, last_nominal_close):
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash entitlement must be finite and nonnegative")
    if not np.isfinite(last_nominal_close) or last_nominal_close <= 0:
        raise ValueError("A positive nominal last close is required")
    return cash / last_nominal_close - 1


def audit(source, staging, reference, output):
    import duckdb

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((staging / "manifest.json").read_text())
    gaps = read_rows(staging / "missing-forward-returns.csv")
    cash = {r["security_id"]: r for r in read_rows(ROOT / "data/cash-acquisitions.csv")}
    stock = {r["security_id"]: r for r in read_rows(ROOT / "data/stock-acquisitions.csv")}
    intervals = suspension_intervals(reference / "exit_gaps.csv")
    c = duckdb.connect()
    c.read_parquet(str(source / "prices.parquet")).create_view("p")
    c.read_parquet(str(source / "membership.parquet")).create_view("m")
    records = []
    for gap in gaps:
        sid = gap["security_id"]
        day = np.datetime64(gap["execution_date"])
        next_day = np.datetime64(gap["next_session"])
        later = c.execute(
            "SELECT date,close,adjusted_close,source FROM p WHERE security_id=? AND date>? ORDER BY date",
            [sid, str(day)],
        ).fetchall()
        known = [v for v in intervals if v["security_id"] == sid]
        later = [r for r in later if not any(
            v["last_trade"] < np.datetime64(r[0]) < v["next_trade"] for v in known
        )]
        prior = c.execute(
            "SELECT close,adjusted_close,source FROM p WHERE security_id=? AND date=?", [sid, str(day)]
        ).fetchone()
        last_member = c.execute("SELECT max(date) FROM m WHERE security_id=?", [sid]).fetchone()[0]
        event = cash.get(sid)
        status = "terminal_outcome_needed"
        cumulative_return = ""
        source_url = ""
        cash_per_share = ""
        if later:
            status = "multi_session_gap"
            cumulative_return = later[0][2] / prior[1] - 1
        if any(v["last_trade"] == day for v in known):
            status = "suspension_requires_frozen_position"
        if event and day <= np.datetime64(event["effective_date"]) <= next_day:
            status = "cash_terms_verified_settlement_pending"
            cash_per_share = event["cash_per_share"]
            cumulative_return = cash_entitlement_return(float(cash_per_share), prior[0])
            source_url = event["source_url"]
        stock_event = stock.get(sid)
        if stock_event and day <= np.datetime64(stock_event["effective_date"]) <= next_day:
            status = "stock_terms_verified_conversion_pending"
            source_url = stock_event["source_url"]
        records.append({
            **gap, "status": status, "last_membership": str(last_member),
            "next_observed_quote": str(later[0][0]) if later else "",
            "cumulative_return_to_next_quote_or_entitlement": cumulative_return,
            "cash_per_original_share": cash_per_share,
            "source_url": source_url,
            "fully_resolved_for_training": False,
        })
    if records:
        with (output / "return-review.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    entitlements = []
    for sid, event in cash.items():
        prior = c.execute(
            "SELECT date,close,source FROM p WHERE security_id=? AND date<=? ORDER BY date DESC LIMIT 1",
            [sid, event["effective_date"]],
        ).fetchone()
        if prior is None:
            continue
        entitlements.append({
            **event, "last_observed_quote": str(prior[0]), "last_nominal_close": prior[1],
            "source": prior[2],
            "entitlement_return_from_last_quote": cash_entitlement_return(float(event["cash_per_share"]), prior[1]),
            "accounting_treatment": "receivable_not_spendable_cash",
            "nominal_price_basis_verified": False,
        })
    if entitlements:
        with (output / "cash-entitlements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entitlements[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(entitlements)
    c.close()
    report = {
        "staging": str(staging),
        "staging_source_sha256": manifest["source_artifact_sha256"],
        "missing_forward_return_observations": len(records),
        "categories": dict(Counter(r["status"] for r in records)),
        "cash_terms_with_primary_sources": len(entitlements),
        "excluded_suspension_quote_rows": manifest.get("excluded_suspension_quote_rows", 0),
        "multi_session_returns_are_not_daily_returns": True,
        "missing_returns_imputed": False,
        "training_ready": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "staging_source_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/final")
    parser.add_argument("--staging", type=Path, default=ROOT / "outputs/training-staging-v2")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/external/sp500-data/terminal-reference")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/terminal-audit")
    args = parser.parse_args()
    audit(args.source, args.staging, args.reference, args.output)


if __name__ == "__main__":
    main()
