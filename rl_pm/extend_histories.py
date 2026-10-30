import argparse
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np

from .build_dataset import digest


ROOT = Path(__file__).resolve().parents[1]


def aligned_extension(reference, candidate, end_exclusive):
    if not reference:
        raise ValueError("A reference history is required")
    common = sorted(set(reference) & set(candidate))[-60:]
    if len(common) < 20:
        raise ValueError("Fewer than 20 overlap observations")
    ratios = np.array([reference[day] / candidate[day] for day in common])
    scale = float(ratios[-1])
    deviation = float(np.max(np.abs(ratios / scale - 1)))
    if not np.isfinite(scale) or scale <= 0 or deviation > .005:
        raise ValueError("Overlap adjustment scale conflicts with the reference")
    last = max(reference)
    extra = {
        day: value * scale for day, value in candidate.items()
        if last < day < end_exclusive and np.isfinite(value) and value > 0
    }
    return extra, scale, len(common), deviation


def run(source, output, rules_path):
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    c = duckdb.connect()
    c.read_parquet(str(source / "prices.parquet")).create_view("original")
    c.execute("CREATE TABLE extensions AS SELECT * FROM original WHERE false")
    reports = []
    with rules_path.open(newline="") as stream:
        rules = list(csv.DictReader(stream))
    for rule in rules:
        sid = rule["security_id"]
        path = ROOT / rule["price_file"]
        reference = dict(c.execute(
            "SELECT date,adjusted_close FROM original WHERE security_id=? ORDER BY date", [sid],
        ).fetchall())
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        candidate = {}
        originals = {}
        for row in rows:
            try:
                day = date.fromisoformat(row["date"])
                value = float(row["adjusted_close"])
            except (ValueError, TypeError):
                continue
            if np.isfinite(value) and value > 0:
                if day in candidate:
                    raise ValueError(f"Duplicate candidate date: {sid} {day}")
                candidate[day] = value
                originals[day] = row
        extra, scale, overlap, deviation = aligned_extension(
            reference, candidate, date.fromisoformat(rule["end_exclusive"]),
        )
        def field(row, name):
            try:
                value = float(row.get(name, ""))
                return value if np.isfinite(value) else None
            except (ValueError, TypeError):
                return None

        inserts = []
        for day, value in sorted(extra.items()):
            row = originals[day]
            inserts.append((
                day, sid, field(row, "open"), field(row, "high"), field(row, "low"),
                field(row, "close"), field(row, "volume"), value,
                field(row, "dividends"), field(row, "stock_splits"),
                "local_continuation:" + row.get("source", path.name),
            ))
        if inserts:
            c.executemany("INSERT INTO extensions VALUES (?,?,?,?,?,?,?,?,?,?,?)", inserts)
        reports.append({
            **rule, "input_sha256": digest(path), "added_observations": len(extra),
            "overlap_observations": overlap, "adjustment_scale": scale,
            "maximum_overlap_deviation": deviation,
            "first_added": str(min(extra)) if extra else None,
            "last_added": str(max(extra)) if extra else None,
            "close_basis": "original_provider_basis_not_independently_verified",
        })
    output.mkdir(parents=True)
    c.sql("""
        SELECT * FROM (SELECT * FROM original UNION ALL SELECT * FROM extensions)
        ORDER BY security_id,date
    """).write_parquet(str(output / "prices.parquet"), compression="zstd")
    total = c.sql("SELECT count(*) FROM extensions").fetchall()[0][0]
    c.close()
    for name in ("membership.parquet", "availability.parquet", "nonquoted-membership-sessions.csv"):
        shutil.copy2(source / name, output / name)
    manifest["parent_manifest_sha256"] = digest(source / "manifest.json")
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["dataset_kind"] = "public_snapshot_plus_dated_local_continuations"
    manifest["daily_price_observations"] += total
    manifest["continuation_histories"] = reports
    manifest["extension_rules_sha256"] = digest(rules_path)
    manifest["execution_alignment_verified"] = False
    manifest["terminal_returns_verified"] = False
    manifest["training_ready"] = False
    manifest["artifact_sha256"] = {name: digest(output / name) for name in manifest["artifact_sha256"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"added_observations": total, "histories": reports, "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/final")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/final-extended")
    parser.add_argument("--rules", type=Path, default=ROOT / "data/continuation-histories.csv")
    args = parser.parse_args()
    run(args.source, args.output, args.rules)


if __name__ == "__main__":
    main()
