import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from .recovery import ROOT, FIELDS, assess, target_windows


AGENT = 'RLPMResearch/1.0'
HOST = 'https://stockanalysis.com'


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None
        self.title = ''
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.in_title = True
        elif tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, value):
        if self.in_title:
            self.title += value
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        elif tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def parse_history(html):
    parser = Tables()
    parser.feed(html)
    for table in parser.tables:
        if not table:
            continue
        headers = [cell.lower().replace('.', '') for cell in table[0]]
        required = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
        if not all(field in headers for field in required):
            continue
        rows = {}
        for cells in table[1:]:
            if len(cells) != len(headers):
                continue
            values = dict(zip(headers, cells))
            try:
                day = datetime.strptime(values['date'], '%b %d, %Y').date().isoformat()
                record = {'date': day}
                for field in ('open', 'high', 'low', 'close', 'volume'):
                    record[field] = float(values[field].replace(',', '').replace('$', ''))
                record['adjusted_close'] = float(values['adj close'].replace(',', '').replace('$', ''))
                if not all(math.isfinite(value) for key, value in record.items() if key != 'date'):
                    continue
                if record['adjusted_close'] <= 0:
                    continue
                rows[day] = record
            except (ValueError, KeyError):
                continue
        if rows:
            return sorted(rows.values(), key=lambda row: row['date']), parser.title.strip()
    raise ValueError('No historical price table in public HTML')


def get_text(url):
    request = Request(url, headers={'User-Agent': AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode('utf-8', errors='replace')


def run(delay, retry):
    output = ROOT / 'data/scraped/stockanalysis'
    output.mkdir(parents=True, exist_ok=True)
    journal = output / 'attempts.jsonl'
    grouped = {}
    for row in target_windows(ROOT / 'outputs/removal-audit.csv'):
        grouped.setdefault(row['ticker'], []).append(row)
    done = set()
    if journal.exists() and not retry:
        for line in journal.read_text().splitlines():
            try:
                done.add(json.loads(line)['ticker'])
            except (ValueError, KeyError):
                continue
    robots = RobotFileParser()
    try:
        robots.parse(get_text(HOST + '/robots.txt').splitlines())
    except (HTTPError, URLError, TimeoutError):
        raise SystemExit('Could not check source access rules; no pages requested.')
    jobs = sorted(set(grouped) - done)
    print(f'{len(jobs):,} historical tickers queued; no ticker-count cap', flush=True)
    counts = Counter()
    for i, symbol in enumerate(jobs, 1):
        url = HOST + '/stocks/' + quote(symbol.lower(), safe='') + '/history/'
        record = dict(ticker=symbol, source=url,
                      retrieved_at=datetime.now(timezone.utc).isoformat(),
                      identity_status='unchecked', settlement_status='unchecked')
        stop = False
        try:
            if not robots.can_fetch(AGENT, url):
                record['status'] = 'robots_disallowed'
            else:
                html = get_text(url)
                html_path = output / (quote(symbol, safe='') + '.html')
                html_path.write_text(html)
                rows, title = parse_history(html)
                destination = output / (quote(symbol, safe='') + '.csv')
                temporary = destination.with_suffix('.csv.part')
                with temporary.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(destination)
                matches = assess(rows, grouped[symbol])
                record.update(status='candidate_scraped' if matches else 'date_mismatch',
                              title=title, rows=len(rows), first_date=rows[0]['date'],
                              last_date=rows[-1]['date'], pre_removal_matches=matches,
                              file=str(destination.relative_to(ROOT)))
        except HTTPError as error:
            record.update(status='not_found' if error.code == 404 else 'http_error', http_code=error.code)
            stop = error.code in (401, 403, 429)
        except (URLError, TimeoutError, ValueError) as error:
            record.update(status='request_failed', error=str(error))
        with journal.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        counts[record['status']] += 1
        if i % 10 == 0 or i == len(jobs) or stop:
            print(f'Processed {i}/{len(jobs)} | ' + ' | '.join(f'{k}: {v}' for k, v in sorted(counts.items())), flush=True)
        if stop:
            print('Source requires access or has rate-limited requests; stopped.', flush=True)
            break
        time.sleep(delay)
    print('Records: data/scraped/stockanalysis/attempts.jsonl', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.delay) or args.delay < 1:
        parser.error('--delay must be at least 1 second')
    run(args.delay, args.retry)


if __name__ == '__main__':
    main()
