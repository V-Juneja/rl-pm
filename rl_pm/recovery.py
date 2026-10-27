import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']


def candidates(symbol):
    return list(dict.fromkeys([symbol, symbol.replace('.', '-')]))


def parse_chart(payload):
    chart = payload.get('chart', {})
    if chart.get('error') or not chart.get('result'):
        raise ValueError('No chart data')
    result = chart['result'][0]
    timestamps = result.get('timestamp', [])
    prices = result['indicators']['quote'][0]
    adjusted = result['indicators'].get('adjclose', [{}])[0].get('adjclose', [])
    rows = []
    for i, timestamp in enumerate(timestamps):
        if i >= len(adjusted) or adjusted[i] is None:
            continue
        value = float(adjusted[i])
        if not math.isfinite(value) or value <= 0:
            continue
        day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        row = {'date': day, 'adjusted_close': value}
        for field in FIELDS[1:-1]:
            series = prices.get(field, [])
            row[field] = series[i] if i < len(series) else None
        rows.append(row)
    if not rows or len({row['date'] for row in rows}) != len(rows):
        raise ValueError('Empty or duplicate price dates')
    return sorted(rows, key=lambda row: row['date']), result.get('meta', {})


def fetch(symbol):
    query = urlencode({'period1': 0, 'period2': int(time.time()), 'interval': '1d'})
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol, safe='') + '?' + query
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=25) as response:
        rows, metadata = parse_chart(json.load(response))
    return rows, metadata, url


def target_windows(path):
    with path.open() as stream:
        return [row for row in csv.DictReader(stream)
                if row['status'] != 'price_window_complete']


def assess(rows, windows):
    dates = {row['date'] for row in rows}
    return sum(bool(window['pre_removal_session']) and window['pre_removal_session'] in dates
               for window in windows)


def run(limit, resume, delay):
    windows = target_windows(ROOT / 'outputs/removal-audit.csv')
    grouped = {}
    for window in windows:
        grouped.setdefault(window['ticker'], []).append(window)
    output = ROOT / 'data/recovery'
    output.mkdir(parents=True, exist_ok=True)
    journal = output / 'attempts.jsonl'
    previous = []
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                previous.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    completed = {(row['historical_ticker'], row['provider_symbol']) for row in previous}
    jobs = [(symbol, candidate) for symbol in sorted(grouped) for candidate in candidates(symbol)
            if not resume or (symbol, candidate) not in completed]
    if limit:
        jobs = jobs[:limit]
    print(f'{len(jobs):,} requests queued', flush=True)
    counts = Counter()
    for index, (symbol, candidate) in enumerate(jobs, 1):
        record = dict(historical_ticker=symbol, provider_symbol=candidate,
                      retrieved_at=datetime.now(timezone.utc).isoformat(),
                      identity_status='unchecked', settlement_status='unchecked')
        stop = False
        try:
            rows, metadata, url = fetch(candidate)
            matches = assess(rows, grouped[symbol])
            record.update(status='candidate_downloaded' if matches else 'date_mismatch',
                          rows=len(rows), first_date=rows[0]['date'], last_date=rows[-1]['date'],
                          removal_windows=len(grouped[symbol]), pre_removal_matches=matches,
                          provider_name=metadata.get('longName', metadata.get('shortName', '')),
                          returned_symbol=metadata.get('symbol'), source=url)
            filename = quote(symbol, safe='') + '__' + quote(candidate, safe='') + '.csv'
            destination = output / filename
            temporary = destination.with_suffix('.csv.part')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(destination)
            record['file'] = str(destination.relative_to(ROOT))
        except HTTPError as error:
            record.update(status='not_found' if error.code == 404 else 'http_error', http_code=error.code)
            stop = error.code in (401, 403, 429)
        except (URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as error:
            record.update(status='request_failed', error=type(error).__name__)
        with journal.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        counts[record['status']] += 1
        if index % 10 == 0 or index == len(jobs) or stop:
            print(f'Processed {index}/{len(jobs)} | ' + ' | '.join(f'{k}: {v}' for k, v in sorted(counts.items())), flush=True)
        if stop:
            print('Provider access/rate limit reached; stopped.')
            break
        time.sleep(delay)
    print('Recovery records: data/recovery/attempts.jsonl')
    print('Candidates stay separate from training data.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    if args.limit < 0 or not math.isfinite(args.delay) or args.delay < 0.5:
        parser.error('Use --limit >= 0 and --delay >= 0.5')
    run(args.limit, not args.retry, args.delay)


if __name__ == '__main__':
    main()
