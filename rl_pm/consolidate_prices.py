"""Combine local price CSVs into a provenance-preserving raw price store."""

import argparse
import csv
from datetime import date
import math
from pathlib import Path
from bisect import bisect_right
from statistics import median
from urllib.parse import quote

from .recovery import FIELDS, ROOT


RAW = ROOT / "data/raw"
OUTPUT = RAW / "combined"
SOURCES = (
    ("raw_yahoo", RAW, 0, "_yahoo_daily.csv"),
    ("tiingo", ROOT / "data/tiingo", 1, None),
    ("yahoo_recovery", ROOT / "data/recovery", 2, None),
    ("stockanalysis", ROOT / "data/scraped/stockanalysis", 3, None),
    ("archive", ROOT / "data/external/tsaustin/prices", 4, "_archive_daily.csv"),
    ("archive_wiki", ROOT / "data/external/wiki/prices", 5, "_archive_daily.csv"),
    ("archive_finsaber", ROOT / "data/external/finsaber/prices", 6, "_archive_daily.csv"),
    ("archive_stooq", ROOT / "data/external/stooq2017/prices", 7, "_archive_daily.csv"),
    ("archive_curated", ROOT / "data/external/sp500-data/rolling/prices", 8, "_archive_daily.csv"),
)

ALIASES = (
    ("BLL", "BALL", "2022-05-10"),
    ("PKI", "RVTY", "2023-05-16"),
    ("ANTM", "ELV", "2022-06-28"),
    ("ABC", "COR", "2023-08-30"),
    ("CTL", "LUMN", "2020-09-18"),
    ("TMK", "GL", "2019-08-09"),
    ("GPS", "GAP", "2024-08-22"),
    ("MMC", "MRSH", "2026-01-14"),
    ("BK", "BNY", "2026-05-21"),
)


def source_files():
    for label, folder, priority, suffix in SOURCES:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            if suffix:
                if not path.name.endswith(suffix):
                    continue
                symbol = path.name[:-len(suffix)]
            elif label == "yahoo_recovery":
                symbol = path.stem.split("__", 1)[0]
            elif label == "tiingo":
                symbol = path.stem.removesuffix("__full")
            else:
                symbol = path.stem
            if symbol:
                yield symbol.upper(), label, priority, path


def read_rows(path):
    rows = {}
    invalid = 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"date", "adjusted_close"} <= set(reader.fieldnames or []):
            return rows, 1
        for row in reader:
            try:
                day = date.fromisoformat(row["date"][:10]).isoformat()
                adjusted = float(row["adjusted_close"])
                if not math.isfinite(adjusted) or adjusted <= 0:
                    raise ValueError
                clean = {"date": day}
                for field in FIELDS[1:]:
                    value = row.get(field, "")
                    if value in (None, ""):
                        clean[field] = ""
                    else:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError
                        clean[field] = format(number, ".12g")
                rows[day] = clean
            except (ValueError, TypeError, KeyError):
                invalid += 1
    return rows, invalid


def values_conflict(first, second):
    for field in FIELDS[1:]:
        a, b = first.get(field, ""), second.get(field, "")
        if not a or not b:
            continue
        try:
            tolerance = (0.5 if field == "volume" else 0.0051)
            if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=tolerance):
                return True
        except ValueError:
            if a != b:
                return True
    return False


def membership_intervals():
    path = ROOT / "data/raw/membership.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        snapshots = sorted((row["date"], set(row["tickers"].split(",")))
                           for row in csv.DictReader(stream))
    active = {}
    result = {}
    for day, members in snapshots:
        for symbol in set(active) - members:
            result.setdefault(symbol, []).append((active.pop(symbol), day))
        for symbol in members - set(active):
            active[symbol] = day
    for symbol, start in active.items():
        result.setdefault(symbol, []).append((start, "9999-12-31"))
    return result


def history_bounds():
    path = ROOT / "data/security-history-bounds.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        return {row["ticker"]: row for row in csv.DictReader(stream)}


def membership_score(rows, intervals):
    starts = [start for start, end in intervals]
    count = 0
    for day in rows:
        position = bisect_right(starts, day) - 1
        if position >= 0 and day < intervals[position][1]:
            count += 1
    return count


def align_adjusted_prices(rows, chosen):
    overlap = sorted(set(rows) & set(chosen))
    if len(overlap) < 20:
        return None, "insufficient_overlap", len(overlap)
    ratios = [float(chosen[day]["adjusted_close"]) / float(rows[day]["adjusted_close"])
              for day in overlap]
    factor = median(ratios)
    if any(abs(value / factor - 1) > 0.01 for value in ratios):
        return None, "inconsistent_adjustments", len(overlap)
    for row in rows.values():
        row["adjusted_close"] = format(float(row["adjusted_close"]) * factor, ".12g")
    return factor, "aligned", len(overlap)


def aligned_edge_extensions(rows, chosen):
    overlap = sorted(set(rows) & set(chosen))
    if len(overlap) < 20:
        return {}, []
    first, last = min(chosen), max(chosen)
    extensions = {}
    reports = []
    for side, anchor, window in (("prefix", first, overlap[:20]),
                                  ("suffix", last, overlap[-20:])):
        if anchor not in window:
            continue
        factor = float(chosen[anchor]["adjusted_close"]) / float(rows[anchor]["adjusted_close"])
        if any(abs(float(chosen[day]["adjusted_close"])
                   / float(rows[day]["adjusted_close"]) / factor - 1) > 0.005
               for day in window):
            continue
        added = 0
        for day, row in rows.items():
            if (side == "prefix" and day < first) or (side == "suffix" and day > last):
                extensions[day] = {**row, "adjusted_close": format(
                    float(row["adjusted_close"]) * factor, ".12g")}
                added += 1
        if added:
            reports.append((side + "_aligned", factor, len(window)))
    return extensions, reports


def combine(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for symbol, label, priority, path in source_files():
        grouped.setdefault(symbol, []).append((priority, label, path))

    cutoffs = {old: change for old, new, change in ALIASES}
    alias_files = {}
    for old, new, change in ALIASES:
        alias_files[old] = [
            (priority + 0.5, label + "_alias", path)
            for priority, label, path in grouped.get(new, [])
        ]
        if alias_files[old]:
            grouped.setdefault(old, []).extend(alias_files[old])

    manifest = []
    conflicts = []
    alignments = []
    intervals = membership_intervals()
    bounds = history_bounds()
    for symbol, files in sorted(grouped.items()):
        chosen = {}
        sources = set()
        invalid_rows = 0
        files = sorted(files, key=lambda item: (item[0], str(item[2])))
        prepared = []
        has_archive = any(label.startswith("archive") for priority, label, path in files)
        for priority, label, path in files:
            rows, invalid = read_rows(path)
            if symbol in cutoffs:
                rows = {day: row for day, row in rows.items()
                        if day < cutoffs[symbol]}
            if symbol in bounds:
                bound = bounds[symbol]
                rows = {day: row for day, row in rows.items()
                        if (not bound["first_date"] or day >= bound["first_date"])
                        and (not bound["last_date"] or day <= bound["last_date"])}
            invalid_rows += invalid
            prepared.append((priority, label, path, rows))
        if has_archive:
            prepared.sort(key=lambda item: (
                -membership_score(item[3], intervals.get(symbol, [])),
                item[0], str(item[2])))
        for priority, label, path, rows in prepared:
            if has_archive and rows and chosen:
                factor, status, overlaps = align_adjusted_prices(rows, chosen)
                alignments.append({"ticker": symbol,
                                   "source": str(path.relative_to(ROOT)),
                                   "status": status, "overlap_dates": overlaps,
                                   "adjusted_close_factor": factor if factor is not None else ""})
                if factor is None:
                    rows, edge_reports = aligned_edge_extensions(rows, chosen)
                    for status, scale, overlaps in edge_reports:
                        alignments.append({"ticker": symbol,
                                           "source": str(path.relative_to(ROOT)),
                                           "status": status, "overlap_dates": overlaps,
                                           "adjusted_close_factor": scale})
                    if not rows:
                        continue
            if rows:
                sources.add(label)
            for day, row in rows.items():
                source_ref = str(path.relative_to(ROOT))
                candidate = {**row, "source": source_ref}
                existing = chosen.get(day)
                if existing is None:
                    chosen[day] = candidate
                elif values_conflict(existing, candidate):
                    conflicts.append({
                        "ticker": symbol,
                        "date": day,
                        "selected_source": existing["source"],
                        "selected_close": existing.get("close", ""),
                        "selected_adjusted_close": existing["adjusted_close"],
                        "other_source": source_ref,
                        "other_close": candidate.get("close", ""),
                        "other_adjusted_close": candidate["adjusted_close"],
                    })

        if not chosen:
            continue
        destination = output / (quote(symbol, safe="") + "_combined_daily.csv")
        temporary = destination.with_suffix(".csv.part")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*FIELDS, "source"])
            writer.writeheader()
            writer.writerows(chosen[day] for day in sorted(chosen))
        temporary.replace(destination)
        dates = sorted(chosen)
        manifest.append({
            "ticker": symbol,
            "rows": len(dates),
            "first_date": dates[0],
            "last_date": dates[-1],
            "source_count": len(sources),
            "sources": ";".join(sorted(sources)),
            "invalid_rows_skipped": invalid_rows,
            "identity_status": "unchecked",
            "settlement_status": "unchecked",
            "file": str(destination.relative_to(ROOT)),
        })

    write_report(output / "manifest.csv", manifest, [
        "ticker", "rows", "first_date", "last_date", "source_count", "sources",
        "invalid_rows_skipped", "identity_status", "settlement_status", "file",
    ])
    write_report(output / "conflicts.csv", conflicts, [
        "ticker", "date", "selected_source", "selected_close", "selected_adjusted_close",
        "other_source", "other_close", "other_adjusted_close",
    ])
    write_report(output / "source-alignment.csv", alignments, [
        "ticker", "source", "status", "overlap_dates", "adjusted_close_factor",
    ])
    return manifest, conflicts


def write_report(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="combined output folder (defaults to data/raw/combined)")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest, conflicts = combine(output)
    total_rows = sum(int(row["rows"]) for row in manifest)
    print(f"Tickers combined: {len(manifest):,}")
    print(f"Daily rows: {total_rows:,}")
    print(f"Conflicting ticker/date pairs: {len(conflicts):,}")
    print(f"Manifest: {output / 'manifest.csv'}")
    print(f"Merged data: {output}")
    print("Ticker identity and terminal settlements remain unverified.")


if __name__ == "__main__":
    main()
