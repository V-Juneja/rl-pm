from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import json
import math
import re
from urllib.parse import quote

from .recovery import FIELDS, ROOT


SOURCE = "https://github.com/Johnbrick123/sp500-data/releases/tag/data"


def base_ticker(label):
    name = re.sub(r"-[0-9]{6}$", "", label)
    return re.sub(r"-([AB])$", r".\1", name)


def membership_keys(label):
    ticker = base_ticker(label)
    aliases = {"DAY": {"CDAY"}, "DOC": {"PEAK"}, "EG": {"RE"}, "TMC": {"TMC.A"},
               "VMRK": {"EQR"}, "ECHO": {"SATS"}, "SKYD": {"PSKY"}}
    return {ticker} | aliases.get(ticker, set())


def identity_segments(intervals, targets):
    groups = defaultdict(list)
    for label, first, end in intervals:
        for ticker in membership_keys(label) & targets:
            groups[ticker].append((label, first.date(), end.date() if end else date.max))
    rules = []
    ambiguous = []
    for ticker, spans in groups.items():
        boundaries = sorted({day for label, first, end in spans for day in (first, end)})
        for first, end in zip(boundaries, boundaries[1:]):
            active = {label for label, left, right in spans if left <= first < right}
            if len(active) > 1:
                ambiguous.append({"membership_ticker": ticker, "from": first.isoformat(),
                                  "through": (end - timedelta(days=1)).isoformat(),
                                  "candidate_securities": ";".join(sorted(active))})
            elif len(active) == 1:
                label = next(iter(active))
                rules.append({"membership_ticker": ticker, "price_ticker": label,
                              "from": first.isoformat(),
                              "through": (end - timedelta(days=1)).isoformat(),
                              "relationship": "external_curated_membership_identity",
                              "source": SOURCE})
    return rules, ambiguous


def run():
    import duckdb

    folder = ROOT / "data/external/sp500-data/rolling"
    output = folder / "prices"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/data-readiness.csv").open(newline="") as stream:
        targets = {row["ticker"] for row in csv.DictReader(stream)
                   if int(row["price_sessions_missing"]) > 0}
    connection = duckdb.connect()
    connection.read_parquet(str(folder / "prices.parquet")).create_view("prices")
    connection.read_csv(str(folder / "no_trade_days.csv")).create_view("no_trade")
    labels = [row[0] for row in connection.execute("SELECT DISTINCT ticker FROM prices").fetchall()]
    selected = [(label,) for label in labels if membership_keys(label) & targets]
    connection.execute("CREATE TEMP TABLE wanted(security_id VARCHAR)")
    connection.executemany("INSERT INTO wanted VALUES (?)", selected)
    query = """
        SELECT p.date, p.ticker, p.open, p.high, p.low, p.close, p.volume,
               p.adj_close, p.source
        FROM prices p
        JOIN wanted w ON p.ticker = w.security_id
        WHERE p.date >= DATE '1995-01-01'
          AND p.source <> 'carried-patch'
          AND isfinite(p.adj_close) AND p.adj_close > 0
          AND isfinite(p.close) AND p.close > 0
          AND NOT EXISTS (
              SELECT 1 FROM no_trade n
              WHERE n.ticker = p.ticker AND n.date = CAST(p.date AS DATE)
          )
        ORDER BY p.ticker, p.date
    """
    cursor = connection.execute(query)
    stream = None
    writer = None
    current = None
    count = 0
    sources = Counter()
    files = Counter()
    previous_day = None
    try:
        while True:
            batch = cursor.fetchmany(10000)
            if not batch:
                break
            for timestamp, label, opening, high, low, close, volume, adjusted, provider in batch:
                day = timestamp.date().isoformat()
                if label != current:
                    if stream is not None:
                        stream.close()
                        temporary.replace(destination)
                    destination = output / (quote(label, safe="") + "_archive_daily.csv")
                    temporary = destination.with_suffix(".csv.part")
                    stream = temporary.open("w", newline="")
                    writer = csv.DictWriter(stream, fieldnames=[*FIELDS, "provider_source"])
                    writer.writeheader()
                    current = label
                    previous_day = None
                if previous_day == day:
                    raise ValueError(f"Duplicate source observation: {label} {day}")
                previous_day = day
                row = dict(zip(FIELDS, [day, opening, high, low, close, volume, adjusted]))
                row = {key: ("" if value is None or isinstance(value, float)
                             and not math.isfinite(value) else value) for key, value in row.items()}
                row["provider_source"] = provider
                writer.writerow(row)
                count += 1
                files[label] += 1
                sources[provider] += 1
        if stream is not None:
            stream.close()
            stream = None
            temporary.replace(destination)
    finally:
        if stream is not None:
            stream.close()
    intervals = connection.execute("SELECT ticker,start,\"end\" FROM read_parquet(?)",
                                   [str(folder / "membership_intervals.parquet")]).fetchall()
    rules, ambiguous = identity_segments(intervals, targets)
    rules = [row for row in rules if row["price_ticker"] in files]
    for row in rules:
        row["price_file"] = str((output / (quote(row["price_ticker"], safe="")
                                          + "_archive_daily.csv")).relative_to(ROOT))
    rules_path = ROOT / "data/membership-price-identities.csv"
    with rules_path.open(newline="") as stream:
        original = list(csv.DictReader(stream))
    original = [row for row in original
                if row["relationship"] != "external_curated_membership_identity"
                or row["membership_ticker"] not in targets]
    with rules_path.open("w", newline="") as stream:
        fields = list(original[0])
        if "price_file" not in fields:
            fields.append("price_file")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(original + rules)
    ambiguity_path = ROOT / "outputs/ambiguous-price-identities.csv"
    if ambiguity_path.exists():
        with ambiguity_path.open(newline="") as stream:
            ambiguous = [row for row in csv.DictReader(stream)
                         if row["membership_ticker"] not in targets] + ambiguous
    with ambiguity_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "membership_ticker", "from", "through", "candidate_securities"
        ])
        writer.writeheader()
        writer.writerows(ambiguous)
    report = {"source": SOURCE, "rows": count, "securities": len(files),
              "sources": dict(sources), "dated_identity_segments": len(rules),
              "ambiguous_segments": len(ambiguous), "carried_prices_imported": 0}
    (folder / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
