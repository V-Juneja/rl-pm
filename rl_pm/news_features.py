import csv
from bisect import bisect_left
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


NEWS_FEATURE_NAMES = (
    "news_log_count_7d",
    "news_log_count_30d",
    "news_sentiment_7d",
    "news_sentiment_30d",
    "news_covered_30d",
)
NEW_YORK = ZoneInfo("America/New_York")


def _timestamp(value, field, line):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"Invalid {field} on CSV line {line}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} needs a timezone on CSV line {line}")
    return parsed.astimezone(timezone.utc)


def build_news_features(events_path, coverage_path, dates, security_ids):
    dates = np.asarray(dates).astype("datetime64[D]")
    ids = np.asarray(security_ids).astype(str)
    positions = {security_id: index for index, security_id in enumerate(ids)}
    events = [[] for _ in ids]
    seen = set()

    with Path(events_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"security_id", "source", "article_id", "published_at", "available_at", "sentiment"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("News events need security_id, article_id, published_at, available_at, sentiment")
        for line, row in enumerate(reader, 2):
            security_id = row["security_id"].strip()
            if security_id not in positions:
                raise ValueError(f"Unknown security_id on news CSV line {line}: {security_id}")
            source = row["source"].strip()
            identity = (security_id, source, row["article_id"].strip())
            if not source:
                raise ValueError(f"Missing source on news CSV line {line}")
            if not identity[1] or identity in seen:
                if identity in seen:
                    continue
                raise ValueError(f"Missing article_id on news CSV line {line}")
            seen.add(identity)
            published = _timestamp(row["published_at"], "published_at", line)
            available = _timestamp(row["available_at"], "available_at", line)
            try:
                sentiment = float(row["sentiment"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid sentiment on news CSV line {line}") from error
            if not np.isfinite(sentiment) or not -1 <= sentiment <= 1:
                raise ValueError(f"Sentiment must lie in [-1, 1] on line {line}")
            events[positions[security_id]].append((max(published, available), sentiment, source))

    coverage = [[] for _ in ids]
    with Path(coverage_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"security_id", "start_date", "end_date", "source"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("News coverage needs security_id, start_date, end_date, source")
        for line, row in enumerate(reader, 2):
            security_id = row["security_id"].strip()
            if security_id not in positions:
                raise ValueError(f"Unknown security_id on coverage CSV line {line}: {security_id}")
            try:
                first = date.fromisoformat(row["start_date"])
                last = date.fromisoformat(row["end_date"])
            except ValueError as error:
                raise ValueError(f"Invalid coverage date on CSV line {line}") from error
            if first > last or not row["source"].strip():
                raise ValueError(f"Invalid news coverage interval on CSV line {line}")
            coverage[positions[security_id]].append((first, last, row["source"].strip()))

    result = np.zeros((len(dates), len(ids), len(NEWS_FEATURE_NAMES)), dtype=np.float32)
    sessions = [date.fromisoformat(str(value)) for value in dates]
    for n, security_id in enumerate(ids):
        company_events = sorted(events[n], key=lambda item: item[0])
        event_times = [item[0] for item in company_events]
        intervals = sorted(coverage[n])
        if not company_events and not intervals:
            continue

        for t in range(1, len(sessions)):
            cutoff_date = sessions[t - 1]
            cutoff = datetime.combine(cutoff_date, time(16), NEW_YORK).astimezone(timezone.utc)
            first_7d = cutoff - timedelta(days=7)
            first_30d = cutoff - timedelta(days=30)
            right = bisect_left(event_times, cutoff + timedelta(microseconds=1))
            left_7d = bisect_left(event_times, first_7d)
            left_30d = bisect_left(event_times, first_30d)

            active_sources = {
                source for first, last, source in intervals
                if first <= cutoff_date <= last
            }
            covered = bool(active_sources)
            result[t, n, 4] = float(covered)
            if not covered:
                continue

            recent_7d = [
                item for item in company_events[left_7d:right]
                if item[2] in active_sources
            ]
            recent_30d = [
                item for item in company_events[left_30d:right]
                if item[2] in active_sources
            ]
            result[t, n, 0] = np.log1p(len(recent_7d))
            result[t, n, 1] = np.log1p(len(recent_30d))
            if recent_7d:
                result[t, n, 2] = np.mean([item[1] for item in recent_7d])
            if recent_30d:
                result[t, n, 3] = np.mean([item[1] for item in recent_30d])

    return result
