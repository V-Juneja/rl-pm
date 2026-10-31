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

The research policy also receives same-date cross-sectional ranks for 21-,
63-, and 252-session momentum and 21-session total and downside volatility.
Ranks are calculated only among securities eligible on that date plus positions
already held. Missing signals receive a neutral rank.

```sh
.venv/bin/python -m rl_pm.features --tickers AAPL MSFT AMZN
```

This preview command uses the project's local combined CSV store.

## Prepare training inputs

```sh
.venv/bin/python scripts/fetch_terminal_reference.py
.venv/bin/python -m rl_pm.build_dataset
```

The builder verifies the source hashes and writes NumPy arrays plus an audit
manifest to `outputs/training-staging/`. Use `--output` to choose a new directory
when rebuilding; existing results are never overwritten. Prices after index
removal remain available. Missing quotes and returns remain missing.

For an execution dated t, features and membership eligibility come from t−1.
The candidate execution is at t's close, with the return measured through the
next session's close. The observed execution quote determines the candidate
tradability flag; future return availability never determines that flag.
Cash earns zero interest in this research staging representation.

`missing-forward-returns.csv` lists buyable observations without a next-session
return. `history-endpoints.csv` lists histories ending before the final session
for terminal-outcome review. Neither report treats a missing quote as a zero
return or a bankruptcy. The staging directory is not an approved training NPZ:
identity, membership timing, execution feasibility, halts, and terminal corporate
actions still require verification. The existing trainer continues to require
its audited dataset contract.

## Terminal-event review

```sh
.venv/bin/python -m rl_pm.build_dataset --output outputs/training-staging-v2
.venv/bin/python -m rl_pm.terminal_audit
```

The pinned reference files include source-reported suspensions extending past
index removal. The builder excludes observations between each reported last
trade and resumption, including unchanged SVB and Signature Bank quotes during
their halts. A resumed multi-session loss is reported over the full gap; it is
not inserted as a daily return or removed from the history.

`outputs/terminal-audit/return-review.csv` classifies every buyable observation
missing a forward return. `cash-entitlements.csv` records cash acquisition
terms with primary sources. These are receivable candidates, not executable prices
or spendable cash. Nominal-price basis and settlement timing still need review.
Abiomed and other deals with contingent rights or stock consideration are not
treated as simple cash exits. No training verification flags are enabled by this
audit.

## Recover longer histories and terminal terms

Some price histories end at exchange or index removal even though the same
shares continued trading OTC. A file endpoint is not itself proof of a merger
or cancellation. The first recovery pass restored 2,089 observations for Delphi,
Mercury Interactive, and Washington Mutual from existing local archives. Each
extension requires at least 20 overlapping quotes with a consistent adjustment
scale, and stops before the documented cancellation or merger date. The
original snapshot remains in place.

```sh
.venv/bin/python -m rl_pm.extend_histories
.venv/bin/python -m rl_pm.build_dataset --source data/raw/final-extended --output outputs/training-staging-v3
.venv/bin/python -m rl_pm.terminal_audit --source data/raw/final-extended --staging outputs/training-staging-v3 --output outputs/terminal-audit-v3
.venv/bin/python -m rl_pm.compile_terminal_events
.venv/bin/python -m rl_pm.recover_terminals --review outputs/terminal-audit-v3/return-review.csv --source data/raw/final-extended --output outputs/terminal-recovery-v3
```

The extended snapshot has 7,220,633 price observations. The event ledger contains
28 cash terms, one stock conversion, and Delphi's cancellation without a
distribution to old common shareholders, with primary-source links. Its
terminal return is recorded at the 2009 cancellation, never at the 2005 index
removal. Cash entitlements remain receivables until payment timing is verified.

Of the 394 remaining buyable observations missing a next-session return, 27
have verified cash terms, one has verified stock-conversion terms, three are
reported suspensions, five are temporary gaps, and 358 still lack verified
terminal terms. An additional public reference matches 168 of those 358, with
dated security identifiers, payout formulas and review flags; these are
candidates, not independently verified outcomes. 190 have no unique nearby
event in that reference.

The reference is
[royelee/delist-detection](https://github.com/royelee/delist-detection), pinned
to the downloaded commit and file hashes in
`data/external/delist-detection/provenance.json`. This project does not run that
repository's code or adopt its estimated zero returns, recovery assumptions, or
confidence labels as verification. Downloaded records and evidence copies remain
under `data/external/`; review reports remain under `outputs/`.

## Run

Research training can run before the independent data audit is complete:

```sh
.venv/bin/python -m rl_pm.research prepare
.venv/bin/python -m rl_pm.research train --updates 50
```

This separate research scenario defaults to zero recovery on unresolved final
exits. Temporary gaps retain the last observed valuation and lock the position
until trading resumes. Known cash consideration uses an explicit assumption
about payment timing and the provider's price basis. These assumptions are
saved with the dataset, checkpoints and run metadata; the verified-data trainer
keeps its existing checks. The cash policy uses a separate Beta allocation so
cash exposure does not vanish as the stock universe grows. Model selection
uses validation years; the final test period requires a separate command.

### Simulated order-execution agent

This is a separate experiment from selecting S&P 500 stocks. A DQN learns how
to split a 100-share parent buy order into market orders, near-touch limit
orders, or waits in ABIDES-NG's simulated limit-order market. Its observation
includes order progress, remaining time, book imbalance, spread, and recent
price movement. Evaluation compares it on held-out simulator seeds with
market-only TWAP, limit-first-then-market, passive-limit-then-market, and
limit-only policies.

Install the pinned simulator into the active project environment, then train
and evaluate:

```sh
PYTHON=.venv/bin/python sh scripts/setup_execution_env.sh
.venv/bin/python -m rl_pm.execution_train --episodes 300 --eval-episodes 50
```

Results go to `outputs/execution-dqn-001/`: `run.json`, `summary.json`,
`evaluation.csv`, `training.csv`, `execution_results.png`, and the model
checkpoint. The simulator checkout is pinned by the setup script and kept under ignored `data/external/`.
The wrapper has a local reset compatibility fix; it does not install Ray/RLlib.

This only tests the policy against synthetic agents and prices generated by the
simulator. It is not historical market replay, evidence of S&P outperformance,
or evidence that a learned order policy would work in a live market. The
simulator fork and execution adapter also need independent review before the
results are treated as research evidence. Background: the [ABIDES-Gym paper](https://arxiv.org/abs/2110.14771)
and the pinned [ABIDES-NG source](https://github.com/GabrieleDiCorato/abides-ng).

### Simulated intraday position management

`rl_pm.intraday_trader` is a second task: a long-only Double DQN can buy one
share, hold, or sell one share in a ten-minute simulated market, up to a
five-share position limit. The state includes inventory, book imbalance,
spread, recent price changes, and elapsed time. It is compared with cash,
buy-and-hold, and a simple momentum rule on separate simulator seeds.

```sh
.venv/bin/python -m rl_pm.intraday_trader --episodes 200 --eval-episodes 30
```

Its checkpoint and outputs are saved under `outputs/intraday-dqn-001/`. This
is a different decision problem from executing a parent order, but it shares
the same synthetic market assumptions. Neither simulation answers whether a
strategy can predict real stock returns.

The larger-exposure sensitivity run uses ten-share actions and a 50-share
limit:

```sh
.venv/bin/python -m rl_pm.intraday_trader --episodes 200 --eval-episodes 30 \
  --position-limit 50 --order-size 10 --output outputs/intraday-scale-dqn-001
```

The same size and training seed can be evaluated under the alternate `rmsc03`
market mix:

```sh
.venv/bin/python -m rl_pm.intraday_trader --episodes 200 --eval-episodes 30 \
  --position-limit 50 --order-size 10 --background-config rmsc03 \
  --output outputs/rmsc03-intraday-dqn-001
```

### Historical S&P 500 exposure control

`rl_pm.index_exposure_rl` asks a narrower historical question using SPY adjusted
prices as a broad S&P 500 total-return proxy: can a Double DQN switch between
0%, 50%, and 100% exposure using only trailing returns, volatility, drawdown,
and trend? The policy observes through close t and earns the next session's
return. It trains on 1996–2012, reports 2013–2018 validation, and holds out
2019 onward for a final test. Five basis points per unit turnover are charged.

```sh
.venv/bin/python -m rl_pm.index_exposure_rl --episodes 250
```

Outputs go to `outputs/index-exposure-dqn-001/`. The comparison includes fixed
cash, 50% and 100% exposure, a 200-day trend filter, and 10% volatility
targeting. This uses the S&P market proxy rather than historical constituent
selection; it is a distinct risk-control experiment, not an estimate of a
constituent-picking strategy. Cash earns zero, and one historical holdout does
not establish a reliable edge.

Recorded 2019–2026 holdout: the first DQN seed returned 90.3% (8.60% CAGR,
17.6% max drawdown) versus 244.8% for buy-and-hold (17.20% CAGR, 33.7% max
drawdown). Fixed 50% exposure returned 92.5% with 18.0% max drawdown, while
the simple volatility-target rule returned 123.3% with 12.7% max drawdown.
A second training seed returned 56.8% (5.94% CAGR, 22.4% max drawdown). The
agent traded far more often than the simple rules. On these runs, RL did not
add value over the simpler risk controls.

### Optional point-in-time news inputs

The research loader accepts `news-events.csv` and `news-coverage.csv` inside the
research data directory when both are present. The event file requires
`security_id,source,article_id,published_at,available_at,sentiment`. Timestamps
must include a timezone; sentiment must be in `[-1, 1]`. `available_at` records
when the dated source made the item available and is used alongside
`published_at` to prevent future information from entering an observation. The
coverage file requires `security_id,start_date,end_date,source` and declares
when a source has complete coverage for that security. Use the stable
`security_id` values in `security_ids.npy`, not a current ticker lookup.

The loader creates rolling article counts and sentiment means over 7 and 30
days, plus a coverage flag. For the row at session t, only items available by
the previous session's 4 p.m. New York close are visible. No news source is
currently in this project’s dataset, so the present training run does not use
news. GDELT 2.0's historical archive starts in February 2015, so it cannot
provide news coverage for the full 1996 backtest; its DOC search API is a recent
search service, not a full-history feed. A source with only partial capture
should not claim complete coverage. See the [GDELT archive overview](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
and [DOC API time range](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/).

Python 3.10+; standard library only for the initial prototype.

```sh
.venv/bin/python -m unittest discover -s tests -v
python3 scripts/fetch_membership.py
python3 -m rl_pm.audit
```

The membership download is exploratory. Price samples do not establish complete
historical or delisted-company coverage.
See [research plan](docs/RESEARCH_PLAN.md) for data sources, limitations and roadmap.
