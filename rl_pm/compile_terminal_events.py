import argparse
import csv
import json
import math
from pathlib import Path

import duckdb

from .build_dataset import digest
from .terminal_audit import cash_entitlement_return


ROOT = Path(__file__).resolve().parents[1]


def compile_events(source, output):
    output.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.read_parquet(str(source / "prices.parquet")).create_view("prices")
    rows, inputs = [], {}
    for filename, rule in (
        ("cash-acquisitions.csv", "cash_receivable"),
        ("stock-acquisitions.csv", "stock_conversion"),
        ("worthless-cancellations.csv", "worthless_cancellation"),
    ):
        path = ROOT / "data" / filename
        inputs[filename] = digest(path)
        with path.open(newline="") as stream:
            terms = list(csv.DictReader(stream))
        for term in terms:
            quote = c.execute(
                "SELECT date,close,source FROM prices WHERE security_id=? AND date<=? ORDER BY date DESC LIMIT 1",
                [term["security_id"], term["effective_date"]],
            ).fetchone()
            if quote is None:
                raise ValueError(f"Missing dated security: {term['security_id']}")
            cash = float(term.get("cash_per_share", term.get("recovery_per_share", 0)))
            ratio = float(term.get("stock_ratio", 0))
            if not math.isfinite(cash) or not math.isfinite(ratio) or cash < 0 or ratio < 0:
                raise ValueError("Consideration must be finite and nonnegative")
            terminal_return = ""
            if rule == "worthless_cancellation":
                terminal_return = -1.0
            elif rule == "cash_receivable" and quote[1] and quote[1] > 0:
                terminal_return = cash_entitlement_return(cash, quote[1])
            rows.append({
                "security_id": term["security_id"], "effective_date": term["effective_date"],
                "event_type": rule, "cash_per_share": cash, "stock_ratio": ratio,
                "acquirer_ticker": term.get("acquirer_ticker", ""),
                "last_observed_quote_date": str(quote[0]), "last_close_native_basis": quote[1],
                "last_quote_source": quote[2],
                "terminal_return_candidate": terminal_return,
                "source_url": term["source_url"], "terms_verified_against_primary_source": True,
                "settlement_date": "", "settlement_cash_spendable": False,
                "price_basis_verified": False, "training_approved": False,
            })
    c.close()
    path = output / "events.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["effective_date"], r["security_id"])))
    manifest = {
        "source_prices_sha256": digest(source / "prices.parquet"), "input_terms_sha256": inputs,
        "events_sha256": digest(path), "events": len(rows),
        "cash_receivables": sum(r["event_type"] == "cash_receivable" for r in rows),
        "stock_conversions": sum(r["event_type"] == "stock_conversion" for r in rows),
        "worthless_cancellations": sum(r["event_type"] == "worthless_cancellation" for r in rows),
        "values_are_market_quotes": False, "training_ready": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/final-extended")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/terminal-events")
    args = parser.parse_args()
    compile_events(args.source, args.output)


if __name__ == "__main__":
    main()
