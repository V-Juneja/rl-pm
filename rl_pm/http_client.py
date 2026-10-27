from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def retry_seconds(value, fallback):
    if value:
        try:
            return max(60, int(value))
        except ValueError:
            try:
                deadline = parsedate_to_datetime(value)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                return max(60, int((deadline - datetime.now(timezone.utc)).total_seconds()) + 1)
            except (ValueError, TypeError, OverflowError):
                pass
    return fallback


def get_text(url, agent):
    backoff = 300
    while True:
        request = Request(url, headers={'User-Agent': agent})
        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode('utf-8', errors='replace')
        except HTTPError as error:
            if error.code not in (429, 503):
                raise
            seconds = retry_seconds(error.headers.get('Retry-After'), backoff)
            resume = datetime.fromtimestamp(time.time() + seconds).astimezone()
            print(f'HTTP {error.code}; paused {seconds:,} seconds. Resume at {resume:%Y-%m-%d %H:%M:%S %Z}.', flush=True)
            remaining = seconds
            while remaining > 0:
                interval = min(60, remaining)
                time.sleep(interval)
                remaining -= interval
            backoff = min(backoff * 2, 3600)
