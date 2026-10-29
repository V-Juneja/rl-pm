from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run():
    import duckdb

    source = ROOT / "data/external/sp500-data/rolling"
    output = ROOT / "data/raw/final"
    output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.read_parquet(str(source / "prices.parquet")).create_view("published_prices")
    connection.read_parquet(str(source / "membership_intervals.parquet")).create_view("published_intervals")
    connection.read_csv(str(source / "no_trade_days.csv")).create_view("reported_no_quote")
    connection.sql("""
        SELECT DISTINCT CAST(date AS DATE) AS date
        FROM published_prices
        WHERE ticker = 'SPY' AND date >= DATE '1996-01-02'
          AND source <> 'carried-patch' AND isfinite(adj_close) AND adj_close > 0
    """).create_view("sessions")
    connection.sql("""
        SELECT CAST(p.date AS DATE) AS date, p.ticker AS security_id,
               p.open, p.high, p.low, p.close, p.volume,
               p.adj_close AS adjusted_close, p.dividends, p.stock_splits, p.source
        FROM published_prices p
        WHERE p.date >= DATE '1995-01-01'
          AND p.source <> 'carried-patch'
          AND isfinite(p.adj_close) AND p.adj_close > 0
          AND isfinite(p.close) AND p.close > 0
          AND NOT EXISTS (
              SELECT 1 FROM reported_no_quote n
              WHERE n.ticker = p.ticker AND n.date = CAST(p.date AS DATE)
          )
    """).create_view("prices")
    duplicate = connection.sql("""
        SELECT security_id,date,count(*) AS n FROM prices
        GROUP BY security_id,date HAVING count(*) > 1 LIMIT 1
    """).fetchall()
    if duplicate:
        raise ValueError(f"Duplicate price observation: {duplicate}")
    connection.sql("""
        SELECT DISTINCT s.date, i.ticker AS security_id
        FROM sessions s JOIN published_intervals i
          ON s.date >= i.start AND (i."end" IS NULL OR s.date < i."end")
    """).create_view("membership")
    connection.sql("""
        SELECT security_id,min(date) AS first_price,max(date) AS last_price
        FROM prices GROUP BY security_id
    """).create_view("price_bounds")
    connection.sql("""
        SELECT m.date,m.security_id,p.security_id IS NOT NULL AS price_available,
               CASE WHEN p.security_id IS NOT NULL THEN 'price_observed'
                    WHEN n.ticker IS NOT NULL THEN 'source_reported_no_quote'
                    WHEN m.date > b.last_price THEN 'after_last_available_price'
                    WHEN m.date < b.first_price THEN 'before_first_available_price'
                    ELSE 'unresolved_price_gap' END AS price_status,
               n.kind AS source_note
        FROM membership m
        LEFT JOIN prices p ON p.date=m.date AND p.security_id=m.security_id
        LEFT JOIN reported_no_quote n ON n.date=m.date AND n.ticker=m.security_id
        LEFT JOIN price_bounds b ON b.security_id=m.security_id
    """).create_view("availability")
    counts = dict(connection.sql("""
        SELECT price_status,count(*) FROM availability GROUP BY price_status
    """).fetchall())
    total = sum(counts.values())
    first = connection.sql("SELECT min(date) FROM sessions").fetchall()[0][0]
    last = connection.sql("SELECT max(date) FROM sessions").fetchall()[0][0]
    initial = connection.sql("""
        SELECT count(*) FROM membership WHERE date=(SELECT min(date) FROM sessions)
    """).fetchall()[0][0]
    prices_count = connection.sql("SELECT count(*) FROM prices").fetchall()[0][0]
    securities = connection.sql("SELECT count(DISTINCT security_id) FROM membership").fetchall()[0][0]
    absent_histories = connection.sql("""
        SELECT count(*) FROM (SELECT DISTINCT security_id FROM membership) m
        WHERE NOT EXISTS (SELECT 1 FROM prices p WHERE p.security_id=m.security_id)
    """).fetchall()[0][0]
    hashes = {}
    for name, query in (
        ("prices.parquet", "SELECT * FROM prices ORDER BY security_id,date"),
        ("membership.parquet", "SELECT * FROM membership ORDER BY date,security_id"),
        ("availability.parquet", "SELECT * FROM availability ORDER BY date,security_id"),
    ):
        path = output / name
        temporary = path.with_suffix(".parquet.part")
        connection.sql(query).write_parquet(str(temporary), compression="zstd")
        temporary.replace(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    connection.sql("""
        SELECT * FROM availability WHERE NOT price_available ORDER BY date,security_id
    """).write_csv(str(output / "nonquoted-membership-sessions.csv"), header=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://github.com/Johnbrick123/sp500-data/releases/tag/data",
        "source_release_metadata": str(source.relative_to(ROOT) / "release.json"),
        "membership_start": str(first),
        "membership_end": str(last),
        "initial_securities": initial,
        "historical_securities": securities,
        "securities_without_any_price_history": absent_histories,
        "daily_price_observations": prices_count,
        "membership_observations": total,
        "price_availability_pct": 100 * counts.get("price_observed", 0) / total,
        "availability_counts": counts,
        "carried_prices_included": 0,
        "artifact_sha256": hashes,
        "security_ids": "external_curated_identifiers_preserved",
        "original_membership_preserved": "data/raw/membership.csv",
        "independently_verified": False,
        "identifiers_verified": False,
        "membership_verified": False,
        "execution_alignment_verified": False,
        "terminal_returns_verified": False,
        "training_ready": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Dataset: {output}")


if __name__ == "__main__":
    run()
