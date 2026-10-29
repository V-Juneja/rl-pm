import csv
import argparse
from datetime import date
import io
import json
import math
from pathlib import Path
from urllib.parse import quote
import zipfile

from .recovery import FIELDS, ROOT


def run_stooq():
    folder = ROOT / "data/external/stooq2017"
    output = folder / "prices"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/data-readiness.csv").open(newline="") as stream:
        targets = {row["ticker"] for row in csv.DictReader(stream)
                   if int(row["price_sessions_missing"]) > 0}
    summaries = {}
    with zipfile.ZipFile(folder / "prices.zip") as archive:
        for name in sorted(archive.namelist()):
            if "/Stocks/" not in "/" + name or not name.endswith(".us.txt"):
                continue
            symbol = Path(name).name[:-7].upper()
            if symbol in ("BF-B", "BRK-B"):
                symbol = symbol.replace("-", ".")
            if symbol not in targets or symbol in summaries:
                continue
            rows = {}
            with archive.open(name) as binary:
                for row in csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig")):
                    try:
                        day = date.fromisoformat(row["Date"]).isoformat()
                        price = float(row["Close"])
                        if not math.isfinite(price) or price <= 0:
                            continue
                    except (KeyError, ValueError, TypeError):
                        continue
                    rows[day] = {**{field: "" for field in FIELDS}, "date": day,
                                 "adjusted_close": format(price, ".12g")}
            if not rows:
                continue
            path = output / (quote(symbol, safe="") + "_archive_daily.csv")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows[day] for day in sorted(rows))
            summaries[symbol] = {"rows": len(rows), "first_date": min(rows),
                                 "last_date": max(rows), "archive_member": name}
    report = {"source": "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
              "price_basis": "split_and_dividend_adjusted",
              "nominal_ohlcv_available": False, "files": summaries}
    (folder / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Stooq candidate histories: {len(summaries):,}", flush=True)
    print(f"Stooq candidate rows: {sum(row['rows'] for row in summaries.values()):,}", flush=True)


def run(source="tsaustin"):
    if source == "stooq2017":
        return run_stooq()
    folder = ROOT / "data/external" / source
    archive_name = {"wiki": "wiki-prices.zip", "tsaustin": "us-historical-prices.zip",
                    "finsaber": "finsaber-prices.zip"}[source]
    symbol_field = "ticker" if source == "wiki" else "symbol"
    adjusted_field = {"wiki": "adj_close", "tsaustin": "close_adjusted",
                      "finsaber": "adjusted_close"}[source]
    dataset = ("marketneutral/quandl-wiki-prices-us-equites" if source == "wiki"
               else "tsaustin/us-historical-stock-prices-with-earnings-data")
    output = folder / "prices"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/data-readiness.csv").open(newline="") as stream:
        targets = {
            row["ticker"] for row in csv.DictReader(stream)
            if int(row["price_sessions_missing"]) > 0
        }
    handles = {}
    writers = {}
    summaries = {}
    invalid = 0
    try:
        with zipfile.ZipFile(folder / archive_name) as archive:
            name = next(name for name in archive.namelist()
                        if name.endswith({"wiki": "WIKI_PRICES.csv",
                                          "tsaustin": "stock_prices_latest.csv",
                                          "finsaber": "prices.csv"}[source]))
            with archive.open(name) as binary:
                reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
                header = next(reader)
                positions = {name: index for index, name in enumerate(header)}
                for index, row in enumerate(reader, 1):
                    symbol = row[positions[symbol_field]]
                    if symbol not in targets:
                        continue
                    try:
                        day = date.fromisoformat(row[positions["date"]]).isoformat()
                        record = {"date": day}
                        for field in FIELDS[1:]:
                            source_field = adjusted_field if field == "adjusted_close" else field
                            number = float(row[positions[source_field]])
                            if not math.isfinite(number):
                                raise ValueError
                            record[field] = number
                        if record["adjusted_close"] <= 0 or record["close"] <= 0:
                            raise ValueError
                    except (ValueError, KeyError, IndexError):
                        invalid += 1
                        continue
                    if symbol not in writers:
                        path = output / (quote(symbol, safe="") + "_archive_daily.csv")
                        handles[symbol] = path.with_suffix(".csv.part").open("w", newline="")
                        writers[symbol] = csv.DictWriter(handles[symbol], fieldnames=FIELDS)
                        writers[symbol].writeheader()
                        summaries[symbol] = {"ticker": symbol, "rows": 0,
                                             "first_date": day, "last_date": day}
                    writers[symbol].writerow(record)
                    summary = summaries[symbol]
                    summary["rows"] += 1
                    summary["first_date"] = min(summary["first_date"], day)
                    summary["last_date"] = max(summary["last_date"], day)
    finally:
        for stream in handles.values():
            stream.close()
    for symbol in writers:
        path = output / (quote(symbol, safe="") + "_archive_daily.csv")
        path.with_suffix(".csv.part").replace(path)
    source_url = ("https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce"
                  if source == "finsaber" else "https://www.kaggle.com/datasets/" + dataset)
    report = {"source": source_url,
              "tickers": len(summaries), "rows": sum(r["rows"] for r in summaries.values()),
              "invalid_rows": invalid, "files": list(summaries.values())}
    (folder / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Archived ticker histories: {report['tickers']:,}", flush=True)
    print(f"Archived daily rows: {report['rows']:,}", flush=True)
    print(f"Invalid rows excluded: {invalid:,}", flush=True)
    print(f"Candidate files: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("tsaustin", "wiki", "finsaber", "stooq2017"), default="tsaustin")
    run(parser.parse_args().source)
