# rl-pm

An experimental RL portfolio research engine for selectively holding, excluding,
and re-entering historical S&P 500 constituents.

## Current status
Historical-price recovery, dated security identifiers, feature calculation, and
an experimental training implementation are present. No model has been trained
on an independently verified dataset yet.

The current downloaded snapshot ends on 2026-10-27 and contains 7,218,544 price
observations. Every one of the 1,228 securities in its historical roster has a
price history. Membership-day price availability is 99.9954%: 50 observations
are flagged by the source as having no quote, and 128 fall after the last
available price. Carried-price rows are excluded.

The original ticker-based membership file is preserved. Its first snapshot has
487 unique tickers; the separately packaged roster distinguishes 500 securities
on that date. These are different membership representations, so their coverage
percentages should not be compared as if they used the same denominator.

Security identity, membership timing, execution alignment, and terminal payouts
still require independent validation before the training gates can be enabled.

## Reproduce the public snapshot

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/fetch_public_snapshot.py
.venv/bin/python -m rl_pm.finalize_data
python3 -m rl_pm.summary
.venv/bin/python scripts/create_journal_images.py
```

The snapshot is sourced from the public
[Johnbrick123/sp500-data release](https://github.com/Johnbrick123/sp500-data/releases/tag/data).
Its original provider labels and dated identifiers are preserved. Published
verification claims belong to that source; this project does not treat them as
an independent verification of its own. Download metadata and SHA-256 hashes
are saved alongside the local dataset.

`data/raw/final/` contains `prices.parquet`, `membership.parquet`,
`availability.parquet`, an exception CSV, and `manifest.json`. Downloaded price
files and generated outputs are excluded from Git.

## Features

`rl_pm/features.py` calculates trailing returns, relative momentum, volatility,
downside volatility, moving-average distances, drawdown, and availability masks.
Features use current and past observations. Features calculated after a close
must be aligned to a later execution boundary by the dataset builder.

```sh
.venv/bin/python -m rl_pm.features --tickers AAPL MSFT AMZN
```

This preview command uses the project's local combined CSV store. The training
dataset builder still needs to align features, eligibility, tradability, and
forward returns, including terminal corporate actions.

## Run
Python 3.10+; standard library only for the initial prototype.

```sh
python3 -m unittest discover -s tests -v
python3 scripts/fetch_membership.py
python3 -m rl_pm.audit
```

The membership download is exploratory. Price samples do not establish complete
historical or delisted-company coverage.
See [research plan](docs/RESEARCH_PLAN.md) for data sources, limitations and roadmap.
