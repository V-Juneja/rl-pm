import csv
from datetime import date
import math


def valid_dates(path):
    dates, invalid = set(), 0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                day = date.fromisoformat(row['date'])
                value = float(row['adjusted_close'])
                if not math.isfinite(value) or value <= 0 or day in dates:
                    raise ValueError('Invalid or duplicate observation')
                dates.add(day)
            except (ValueError, KeyError, TypeError):
                invalid += 1
    return dates, invalid

