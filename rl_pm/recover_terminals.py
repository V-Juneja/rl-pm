import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import duckdb


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "royelee/delist-detection"
REFERENCE_COMMIT = "806078a02b02f9b4d01ddcd7eb7e8fc4eb8db5de"
FILES = (
    "output/contract/delistings.csv", "output/contract/payout_legs.csv",
    "output/contract/security_history.csv", "output/contract/price_requests.csv",
    "output/delistings.csv", "output/securities.csv", "output/observation_map.csv",
    "output/review.csv", "output/run_manifest.json", "output/scorecard.json",
    "data/diagnosis_truth.csv", "data/diagnosis_truth_legs.csv", "LICENSE",
)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_url(url):
    with urlopen(Request(url, headers={"User-Agent": "RLPMResearch/1.0"}), timeout=60) as response:
        return response.read()


def fetch_reference(folder):
    folder.mkdir(parents=True, exist_ok=True)
    provenance_path = folder / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        commit = provenance["commit"]
    else:
        commit = REFERENCE_COMMIT
        provenance = {"repository": REPOSITORY, "commit": commit, "files": {}}
    for name in FILES:
        path = folder / name
        expected = provenance["files"].get(name, {}).get("sha256")
        if path.exists() and expected:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Cached reference changed: {name}")
            continue
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{name}"
        content = read_url(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        provenance["files"][name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def base_ticker(security_id):
    return re.sub(r"-\d{6}$", "", security_id).replace("-", ".")


def match_event(gap, events, tolerance=10):
    ticker = base_ticker(gap["security_id"])
    day = date.fromisoformat(gap["execution_date"])
    found = [
        event for event in events
        if event["last_trade_date"]
        and base_ticker(event["ticker"]) == ticker
        and abs((date.fromisoformat(event["last_trade_date"]) - day).days) <= tolerance
    ]
    return found[0] if len(found) == 1 else None


def write_rows(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch_evidence(folder, provenance, matches):
    commit = provenance["commit"]
    tree_path = folder / "tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(read_url(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{commit}?recursive=1"))
    tree = json.loads(tree_path.read_text())["tree"]
    ids = {row["sec_id"] for row in matches}
    ciks = {int(row["cik"]) for row in matches if row["cik"].isdigit()}
    wanted = []
    for entry in tree:
        name = entry["path"]
        filename = name.split("/")[-1]
        report = name.startswith("data/diagnosis/reports/") and any(filename.startswith(sid + "_") for sid in ids)
        text = (
            name.startswith("tests/fixtures/golden/text/")
            and filename.split("_")[0].isdigit()
            and int(filename.split("_")[0]) in ciks
        )
        if report or text:
            wanted.append(name)

    def fetch(name):
        path = folder / name
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{name}"
        if not path.exists():
            content = read_url(url)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = provenance["files"].get(name, {}).get("sha256")
        if previous and value != previous:
            raise ValueError(f"Evidence hash mismatch: {name}")
        return name, {"url": url, "sha256": value}

    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, metadata in executor.map(fetch, wanted):
            provenance["files"][name] = metadata
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return wanted


def run(folder, review, source, output):
    provenance = fetch_reference(folder)
    events = read_rows(folder / "output/delistings.csv")
    contracts = {
        (r["sec_id"], r["last_trade_date"]): r
        for r in read_rows(folder / "output/contract/delistings.csv")
    }
    gaps = [r for r in read_rows(review) if r["status"] == "terminal_outcome_needed"]
    matched = [(gap, match_event(gap, events)) for gap in gaps]
    evidence = fetch_evidence(folder, provenance, [event for _, event in matched if event])
    c = duckdb.connect()
    c.read_parquet(str(source / "prices.parquet")).create_view("prices")
    candidates, unmatched, excerpts = [], [], []
    for gap, event in matched:
        if event is None:
            unmatched.append({
                "security_id": gap["security_id"], "execution_date": gap["execution_date"],
                "next_session": gap["next_session"],
                "needed": "issuer identity and terminal consideration",
                "search_query": f'{base_ticker(gap["security_id"])} {gap["execution_date"][:4]} acquisition merger per share SEC',
            })
            continue
        contract = contracts.get((event["sec_id"], event["last_trade_date"]), {})
        prior = c.execute(
            "SELECT close,source FROM prices WHERE security_id=? AND date=?",
            [gap["security_id"], gap["execution_date"]],
        ).fetchone()
        reference_close = float(event["last_trade_close"]) if event["last_trade_close"] else None
        close_agrees = bool(prior and reference_close and abs(prior[0] / reference_close - 1) <= .01)
        accession = event["delist_filing_accession"]
        cik = event["cik"]
        filing_url = (
            f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-", "")}/{accession}.txt'
            if cik.isdigit() and accession else ""
        )
        paths = [name for name in evidence if Path(name).name.startswith(event["sec_id"] + "_")]
        urls = set()
        for name in paths:
            text = (folder / name).read_text()
            urls.update(re.findall(r'https://(?:www\.)?sec\.gov/[^\s<>\)\]\"`]+', text))
            cash = contract.get("cash_per_share", "")
            if cash:
                number = float(cash)
                pattern = r'\$\s*' + re.escape(f"{number:,.2f}").replace(r"\,", ",?")
                for hit in list(re.finditer(pattern, text))[:2]:
                    excerpts.append({
                        "security_id": gap["security_id"], "evidence_file": str(folder / name),
                        "excerpt": " ".join(text[max(0, hit.start()-120):hit.end()+200].split()),
                        "primary_evidence_verified": False,
                    })
        rule = contract.get("value_rule", "")
        cash = contract.get("cash_per_share", "")
        return_candidate = ""
        if rule == "cash" and cash and close_agrees:
            return_candidate = float(cash) / prior[0] - 1
        candidates.append({
            "security_id": gap["security_id"], "execution_date": gap["execution_date"],
            "company_name": event["resolved_name"], "issuer_cik": cik,
            "reference_sec_id": event["sec_id"], "reference_last_trade": event["last_trade_date"],
            "last_trade_date_matches": gap["execution_date"] == event["last_trade_date"],
            "last_close_agrees_with_reference": close_agrees,
            "local_last_close": prior[0] if prior else "", "local_price_source": prior[1] if prior else "",
            "reference_last_close": reference_close if reference_close else "",
            "value_rule": rule, "cash_per_share": cash,
            "stock_ratio": contract.get("stock_ratio", ""),
            "successor_sec_id": contract.get("successor_sec_id", ""),
            "acquirer_ticker": contract.get("price_ticker", ""),
            "acquirer_price_date": contract.get("price_date", ""),
            "cash_entitlement_return_candidate": return_candidate,
            "reference_verdict": contract.get("verdict", ""),
            "reference_terms_source": contract.get("terms_source", ""),
            "reference_terms_gate": contract.get("terms_gate", ""),
            "reference_review_flags": event["review_flags"],
            "form25_source_url": filing_url,
            "primary_source_urls_from_reports": ";".join(sorted(urls)),
            "evidence_reports": ";".join(str(folder / name) for name in paths),
            "primary_evidence_verified": False,
            "fully_resolved_for_training": False,
        })
    c.close()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "payout-candidates.csv", candidates)
    write_rows(output / "still-unmatched.csv", unmatched)
    write_rows(output / "evidence-excerpts.csv", excerpts)
    summary = {
        "source_repository": f"https://github.com/{REPOSITORY}",
        "source_commit": provenance["commit"],
        "terminal_cases_examined": len(gaps),
        "matched_payout_records": len(candidates), "unmatched": len(unmatched),
        "candidate_value_rules": dict(Counter(r["value_rule"] for r in candidates)),
        "exact_last_trade_date_matches": sum(r["last_trade_date_matches"] for r in candidates),
        "last_close_agrees_with_reference": sum(r["last_close_agrees_with_reference"] for r in candidates),
        "downloaded_evidence_documents": len(evidence),
        "primary_filings_linked": sum(bool(r["primary_source_urls_from_reports"]) for r in candidates),
        "new_fully_verified_training_outcomes": 0,
        "training_ready": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/external/delist-detection")
    parser.add_argument("--review", type=Path, default=ROOT / "outputs/terminal-audit/return-review.csv")
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/final")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/terminal-recovery")
    args = parser.parse_args()
    run(args.reference, args.review, args.source, args.output)


if __name__ == "__main__":
    main()
