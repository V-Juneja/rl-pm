import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, stock_features
from .terminal_audit import apply_suspensions, suspension_intervals


ROOT = Path(__file__).resolve().parents[1]


def forward_returns(prices):
    prices = np.asarray(prices, dtype=np.float64)
    result = np.full(prices.shape, np.nan)
    valid = (
        np.isfinite(prices[:-1]) & (prices[:-1] > 0)
        & np.isfinite(prices[1:]) & (prices[1:] > 0)
    )
    np.divide(prices[1:], prices[:-1], out=result[:-1], where=valid)
    result[:-1][valid] -= 1
    return result


def lagged_features(prices, benchmark):
    values = stock_features(prices, benchmark)
    result = np.zeros_like(values)
    result[1:] = values[:-1]
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(source, output, start, exit_gaps=None):
    import duckdb

    if output.exists():
        raise ValueError(f"Output already exists; choose a new --output: {output}")
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Source hash mismatch: {name}")

    connection = duckdb.connect()
    connection.read_parquet(str(source / "prices.parquet")).create_view("prices")
    connection.read_parquet(str(source / "membership.parquet")).create_view("membership")
    dates = connection.sql("""
        SELECT DISTINCT date FROM prices WHERE security_id='SPY' ORDER BY date
    """).fetchnumpy()["date"].astype("datetime64[D]")
    ids = connection.sql("""
        SELECT DISTINCT security_id FROM membership ORDER BY security_id
    """).fetchnumpy()["security_id"].astype(str)
    if len(dates) < 3 or len(ids) == 0:
        raise ValueError("Insufficient sessions or securities")
    left = int(np.searchsorted(dates, np.datetime64(start)))
    right = len(dates) - 1
    if left < 1 or right <= left:
        raise ValueError("Start must leave a prior observation and a forward session")

    full_shape = (len(dates), len(ids))
    prices = np.full(full_shape, np.nan)
    members = np.zeros(full_shape, dtype=bool)
    rows = connection.sql("""
        SELECT p.date,p.security_id,p.adjusted_close FROM prices p
        WHERE EXISTS (SELECT 1 FROM membership m WHERE m.security_id=p.security_id)
    """).fetchnumpy()
    day = rows["date"].astype("datetime64[D]")
    ti = np.searchsorted(dates, day)
    ni = np.searchsorted(ids, rows["security_id"].astype(str))
    inside = ti < len(dates)
    inside[inside] &= dates[ti[inside]] == day[inside]
    prices[ti[inside], ni[inside]] = rows["adjusted_close"][inside]
    del rows, day, ti, ni, inside
    if exit_gaps is None or not exit_gaps.exists():
        raise ValueError("An --exit-gaps source is required to exclude reported suspension quotes")
    excluded_suspension_quotes = apply_suspensions(
        prices, dates, ids, suspension_intervals(exit_gaps),
    )
    rows = connection.sql("SELECT date,security_id FROM membership").fetchnumpy()
    day = rows["date"].astype("datetime64[D]")
    ti = np.searchsorted(dates, day)
    if (ti >= len(dates)).any() or not np.array_equal(dates[ti], day):
        raise ValueError("Membership contains a date outside the benchmark calendar")
    members[ti, np.searchsorted(ids, rows["security_id"].astype(str))] = True
    del rows, day, ti
    benchmark = connection.sql("""
        SELECT adjusted_close FROM prices WHERE security_id='SPY' ORDER BY date
    """).fetchnumpy()["adjusted_close"]
    connection.close()

    output.mkdir(parents=True)
    shape = (right - left, len(ids))
    np.save(output / "dates.npy", dates[left:right])
    np.save(output / "security_ids.npy", ids)
    eligible = members[left - 1:right - 1].copy()
    tradable = np.isfinite(prices[left:right])
    returns = forward_returns(prices)[left:right]
    np.save(output / "eligible.npy", eligible)
    np.save(output / "tradable.npy", tradable)
    np.save(output / "returns.npy", returns)
    np.save(output / "benchmark.npy", forward_returns(benchmark)[left:right])
    np.save(output / "cash_returns.npy", np.zeros(shape[0]))
    features = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype=np.float32,
        shape=(*shape, len(FEATURE_NAMES)),
    )
    for index, security_id in enumerate(ids):
        features[:, index] = lagged_features(prices[:, index], benchmark)[left:right]
        if (index + 1) % 100 == 0 or index + 1 == len(ids):
            print(f"Features {index + 1:,}/{len(ids):,}", flush=True)
    features.flush()
    del features

    missing = eligible & tradable & ~np.isfinite(returns)
    with (output / "missing-forward-returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["execution_date", "security_id", "next_session"])
        for t, n in zip(*np.nonzero(missing)):
            writer.writerow([dates[left + t], ids[n], dates[left + t + 1]])
    with (output / "history-endpoints.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["security_id", "last_price", "last_membership", "review_status"])
        for n, security_id in enumerate(ids):
            observed = np.flatnonzero(np.isfinite(prices[:, n]))
            membership = np.flatnonzero(members[:, n])
            if len(observed) and observed[-1] < len(dates) - 1:
                writer.writerow([
                    security_id, dates[observed[-1]],
                    dates[membership[-1]] if len(membership) else "",
                    "terminal_outcome_not_verified",
                ])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_staging_only",
        "source_manifest_sha256": digest(source / "manifest.json"),
        "source_artifact_sha256": manifest["artifact_sha256"],
        "sessions": shape[0], "securities": shape[1],
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "start": str(dates[left]), "end": str(dates[right - 1]),
        "feature_timing": "through previous session close",
        "eligibility_timing": "previous session membership; effective dates not announcement dates",
        "execution_assumption": "current close; closing execution feasibility not verified",
        "return_timing": "current close to next session close, adjusted prices",
        "tradability": "observed current price only; halt and liquidity checks pending",
        "cash_assumption": "zero interest",
        "post_removal_prices_preserved": True,
        "missing_prices_filled": False,
        "excluded_suspension_quote_rows": excluded_suspension_quotes,
        "exit_gaps_sha256": digest(exit_gaps),
        "suspension_source_independently_verified": False,
        "missing_forward_returns_for_buyable": int(missing.sum()),
        "eligible_without_execution_price": int((eligible & ~tradable).sum()),
        "identifiers_verified": manifest.get("identifiers_verified", False),
        "membership_verified": manifest.get("membership_verified", False),
        "execution_alignment_verified": False,
        "terminal_returns_verified": False,
        "features_point_in_time": True,
        "training_ready": False,
    }
    report["artifact_sha256"] = {
        path.name: digest(path) for path in sorted(output.glob("*.npy"))
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {shape[0]:,} sessions × {shape[1]:,} securities × {len(FEATURE_NAMES)} features")
    print(f"Buyable observations missing next-session returns: {missing.sum():,}")
    print(f"Dataset: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/final")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training-staging")
    parser.add_argument("--start", default="1996-01-02")
    parser.add_argument("--exit-gaps", type=Path, default=ROOT / "data/external/sp500-data/terminal-reference/exit_gaps.csv")
    args = parser.parse_args()
    run(args.source, args.output, args.start, args.exit_gaps)


if __name__ == "__main__":
    main()
