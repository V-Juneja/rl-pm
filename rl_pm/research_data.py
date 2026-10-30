import csv
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

from .build_dataset import digest
from .terminal_audit import apply_suspensions, suspension_intervals


ROOT = Path(__file__).resolve().parents[1]


def valuation_returns(quotes):
    quotes = np.asarray(quotes, dtype=np.float64)
    valid = np.isfinite(quotes) & (quotes > 0)
    positions = np.where(valid, np.arange(len(quotes)), -1)
    positions = np.maximum.accumulate(positions)
    marks = np.full(len(quotes), np.nan)
    seen = positions >= 0
    marks[seen] = quotes[positions[seen]]
    returns = np.full(len(quotes) - 1, np.nan)
    known = np.isfinite(marks[:-1]) & np.isfinite(marks[1:])
    returns[known] = marks[1:][known] / marks[:-1][known] - 1
    return returns


def prepare(staging, source, output, unknown_recovery):
    if not 0 <= unknown_recovery <= 1:
        raise ValueError("Unknown recovery must lie between zero and one")
    if output.exists():
        raise ValueError(f"Research cache already exists: {output}")
    metadata = json.loads((staging / "manifest.json").read_text())
    for name, expected in metadata["artifact_sha256"].items():
        if digest(staging / name) != expected:
            raise ValueError(f"Staging hash mismatch: {name}")
    for name, expected in metadata["source_artifact_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Price source does not match staging: {name}")
    dates = np.load(staging / "dates.npy")
    ids = np.load(staging / "security_ids.npy")
    c = duckdb.connect()
    c.read_parquet(str(source / "prices.parquet")).create_view("p")
    next_date = c.execute(
        "SELECT min(date) FROM p WHERE security_id='SPY' AND date>?", [str(dates[-1])],
    ).fetchone()[0]
    if next_date is None:
        raise ValueError("Missing final forward session")
    calendar = np.concatenate([dates, np.array([next_date], dtype="datetime64[D]")])
    quotes = np.full((len(calendar), len(ids)), np.nan)
    nominal = np.full_like(quotes, np.nan)
    rows = c.execute("""
        SELECT date,security_id,adjusted_close,close FROM p
        WHERE date>=? AND date<=?
          AND security_id IN (SELECT DISTINCT security_id FROM read_parquet(?))
    """, [str(calendar[0]), str(calendar[-1]), str(source / "membership.parquet")]).fetchnumpy()
    ti = np.searchsorted(calendar, rows["date"].astype("datetime64[D]"))
    ni = np.searchsorted(ids, rows["security_id"].astype(str))
    valid = (ti < len(calendar)) & (ni < len(ids))
    valid[valid] &= calendar[ti[valid]] == rows["date"].astype("datetime64[D]")[valid]
    quotes[ti[valid], ni[valid]] = rows["adjusted_close"][valid]
    raw_close = np.ma.filled(rows["close"], np.nan)
    nominal[ti[valid], ni[valid]] = raw_close[valid]
    c.close()
    del rows, ti, ni, valid
    gap_path = ROOT / "data/external/sp500-data/terminal-reference/exit_gaps.csv"
    if digest(gap_path) != metadata["exit_gaps_sha256"]:
        raise ValueError("Suspension reference differs from staging")
    apply_suspensions(quotes, calendar, ids, suspension_intervals(gap_path))
    returns = np.full((len(dates), len(ids)), np.nan)
    settlements = np.zeros(returns.shape, dtype=bool)
    assumed = np.zeros(returns.shape, dtype=bool)
    terms = {}
    term_hashes = {}
    for filename, kind in (("cash-acquisitions.csv", "cash"), ("worthless-cancellations.csv", "zero")):
        path = ROOT / "data" / filename
        term_hashes[filename] = digest(path)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                terms[row["security_id"]] = {**row, "kind": kind}
    events = []
    temporary_intervals = 0
    for n, sid in enumerate(ids):
        q = quotes[:, n]
        observed = np.flatnonzero(np.isfinite(q) & (q > 0))
        if not len(observed):
            continue
        returns[:, n] = valuation_returns(q)
        last = int(observed[-1])
        temporary_intervals += int(np.isnan(q[observed[0]:last]).sum())
        if last == len(calendar) - 1:
            continue
        event = terms.get(str(sid))
        event_step = last
        recovery = unknown_recovery
        kind = "assumed_terminal_recovery"
        url = ""
        if event:
            effective = np.datetime64(event["effective_date"])
            settlement = max(last + 1, int(np.searchsorted(calendar, effective)))
            day_gap = int((effective - calendar[last]).astype(int))
            if 0 <= day_gap <= 31 and settlement < len(calendar):
                if event["kind"] == "zero":
                    recovery = 0.0
                    kind = "documented_zero_cancellation"
                    event_step = settlement - 1
                    url = event["source_url"]
                elif np.isfinite(nominal[last, n]) and nominal[last, n] > 0:
                    recovery = float(event["cash_per_share"]) / nominal[last, n]
                    kind = "cash_terms_with_assumed_payment_timing_and_price_basis"
                    event_step = settlement - 1
                    url = event["source_url"]
        returns[event_step, n] = recovery - 1
        returns[event_step + 1:, n] = 0.0
        settlements[event_step, n] = True
        assumed[event_step, n] = kind != "documented_zero_cancellation"
        events.append({
            "security_id": str(sid), "last_observed_quote": str(calendar[last]),
            "return_interval_start": str(dates[event_step]),
            "accounting_date": str(calendar[event_step + 1]), "event_type": kind,
            "recovery_multiple": recovery, "source_url": url,
            "verified_training_outcome": False,
        })
    original_tradable = np.load(staging / "tradable.npy")
    if not np.array_equal(original_tradable, np.isfinite(quotes[:-1])):
        raise ValueError("Research changed the observable tradability mask")
    output.mkdir(parents=True)
    for name, array in (("returns", returns), ("cash_settlements", settlements), ("assumed_exits", assumed)):
        np.save(output / (name + ".npy"), array)
    with (output / "events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)
    assumptions = {
        "mode": "research_scenario_not_verified_backtest",
        "unknown_terminal_recovery_multiple": unknown_recovery,
        "unknown_terminal_rule": "assumed recovery at first missing session after final observed quote",
        "temporary_gap_valuation": "last observed adjusted value; position cannot trade during the gap",
        "cash_event_timing": "assumed payable at first session on or after legal event, after last quote",
        "cash_event_price_basis": "provider native close assumed as-traded",
        "cash_interest": 0.0,
        "initial_holdings": "cash; historical members eligible, with 500 securities in initial roster",
        "retained_removed_stocks": "held securities remain controllable when quoted; no new entry after full exit",
        "terminal_events": len(events), "temporary_missing_quote_sessions": temporary_intervals,
        "documented_zero_cancellations": sum(e["event_type"] == "documented_zero_cancellation" for e in events),
        "cash_terms_used": sum(e["event_type"].startswith("cash_terms") for e in events),
        "assumed_unknown_exits": sum(e["event_type"] == "assumed_terminal_recovery" for e in events),
    }
    report = {
        "staging": str(staging.resolve()), "source": str(source.resolve()),
        "staging_manifest_sha256": digest(staging / "manifest.json"),
        "staging_artifact_sha256": metadata["artifact_sha256"],
        "term_input_sha256": term_hashes,
        "artifact_sha256": {p.name: digest(p) for p in output.glob("*.npy")},
        "events_sha256": digest(output / "events.csv"),
        "research_assumptions": assumptions,
        "source_verification": {key: metadata.get(key) for key in (
            "identifiers_verified", "membership_verified", "execution_alignment_verified", "terminal_returns_verified",
        )},
        "training_ready": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(assumptions, indent=2), flush=True)


def load_market(path):
    from .allocation_policy import AllocationPolicy
    from .research_environment import ResearchPortfolio
    from .train import Market

    class ResearchMarket(Market):
        environment_class = ResearchPortfolio
        model_class = AllocationPolicy
        model_architecture = "beta-cash-dirichlet-stocks-v1"

        def __init__(self):
            self.path = path
            self.metadata = json.loads((path / "manifest.json").read_text())
            staging = Path(self.metadata["staging"])
            if digest(staging / "manifest.json") != self.metadata["staging_manifest_sha256"]:
                raise ValueError("Staging manifest changed")
            for name, expected in self.metadata["staging_artifact_sha256"].items():
                if digest(staging / name) != expected:
                    raise ValueError(f"Staging artifact changed: {name}")
            for name, expected in self.metadata["artifact_sha256"].items():
                if digest(path / name) != expected:
                    raise ValueError(f"Research artifact changed: {name}")
            if digest(path / "events.csv") != self.metadata["events_sha256"]:
                raise ValueError("Research event ledger changed")
            self.digest = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
            self.dates = np.load(staging / "dates.npy")
            self.ids = np.load(staging / "security_ids.npy")
            self.features = np.load(staging / "features.npy", mmap_mode="r")
            for name in ("eligible", "tradable", "benchmark", "cash_returns"):
                setattr(self, name, np.load(staging / (name + ".npy")))
            for name in ("returns", "cash_settlements", "assumed_exits"):
                setattr(self, name, np.load(path / (name + ".npy")))
            self.validate()

    return ResearchMarket()
