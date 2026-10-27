# Research plan

## Question
Can a long-only RL portfolio manager improve risk-adjusted outcomes by excluding
and re-entering stocks from the contemporaneous S&P 500 universe, after costs?
This does not predict or control the index committee's membership decisions.

## Data decision — 2026-10-26
Downloaded candidate membership snapshots from
https://github.com/fja05680/sp500 (local data/raw/membership.csv).
Audit: 2,720 snapshots, 1,209 distinct ticker strings, 1996-01-02 to 2026-08-18.
These are change snapshots, not daily price rows. Counts range from 487 to 507.
The maintainer flags incomplete early coverage. Tickers are not stable identifiers.
This file is exploratory and does not establish survivorship-free coverage.
No market training is permitted until the data requirements below are satisfied.

Preferred commercial candidate: Norgate US Platinum ($630/year at research date),
with active/delisted securities and historical membership; Diamond extends history.
https://norgatedata.com/stockmarketpackages.php
https://norgatedata.com/data-package-faq.php
https://norgatedata.com/data-content-tables.php
Supported local Python access uses Windows, so this Mac needs a separate supported
data host or an alternative licensed source. Verify license/export rights before
building any SaaS that redistributes data. No purchase has been made.
More history is not automatically better: start with a well-audited modern window.

## Required normalized data
- Stable security ID, ticker history, effective membership intervals.
- Daily raw OHLCV, corporate actions, distributions and terminal settlement returns.
- Observation/availability timestamps and source provenance.
- Explicit trading calendar and missing/suspended/delisted distinctions.
- Price coverage for every held security through final liquidation.
Never remove a company because later prices are missing; fail the run for review.
Index membership is effective-date eligibility, not advance knowledge of a change.
For removals, retain the position's settlement accounting even when no longer buyable.

## Execution and learning design
Observe through close t; execute no earlier than open t+1. Daily data, weekly actions.
Use price-relative historical features initially; no current fundamentals in past dates.
Long-only, cash permitted, no leverage; add concentration and turnover limits.
Use shared per-security policy scoring with an eligibility mask and cash output,
so the action space does not depend on today's tickers. Holdings must be in state.
PPO is a candidate, not yet implemented. Compare with equal weight, cash,
and a properly licensed total-return S&P 500 benchmark.
Reward: portfolio log return after costs, with explicitly tuned risk/turnover terms.
Evaluate raw returns and drawdowns independently of the training reward.

Chronological train/validation/test windows; purge overlapping feature/label windows.
Fit preprocessing only on training. Select on validation, freeze final test once.
Run multiple seeds and rolling walk-forward windows. Report fees/slippage sensitivity,
turnover, exposure, concentration, drawdown and uncertainty, not just best returns.
The preliminary generated-data demonstration has been removed. No RL agent is
currently implemented; training will use validated market data.

## Delivery milestones and PIXL
1. Data ingestion, provenance, coverage checks and identifier mapping.
2. Validated accounting environment and baseline backtests.
3. RL training, frozen evaluation, saved configurations/checkpoints.
4. Experiment dashboard: equity curves, holdings, exits/re-entries, data quality.

Target T3 Grid for the completed system, subject to reviewer judgment.
T4 only if delivered architecture and depth warrant it; model terminology alone does not.
https://pixl.hackclub.com/docs/energy
No trial required: own idea. Project remains unregistered until its actual data and
submission fields can be completed accurately. No hours or journal claims created.

## Delisted-company data route (revised)
Sharadar Prices is the preferred Mac-compatible API candidate to evaluate:
- stocks: active and delisted OHLCV, history to December 1997.
- tickers: permanent identifiers, related tickers, delisted flags and coverage dates.
- actions: ticker changes, mergers, dividends and delisting reasons, from 1998.
- sp500: historical membership events/effective dates and quarterly snapshots from 1998.
Sources: https://sharadar.com/docs/stocks , https://sharadar.com/docs/tickers ,
https://sharadar.com/docs/actions , https://sharadar.com/docs/sp500 .

Requires a paid subscription/API key; access has not been obtained. This is a
procurement and integration route, not a claim that coverage is already resolved.
Use January 1998 onward as a candidate common start, subject to per-security audit;
keep the free 1996 membership file for cross-checks rather than invent missing prices.
Download all four tables with years=full using documented authenticated bulk access.
Join identifiers and action history before matching historical membership symbols.
Verify terminal settlement proceeds separately: a delisting reason does not establish
an investor's final return. Missing proceeds need source-backed reconciliation,
including filings where necessary. Never assume every delisting is a 100% loss.
Audit membership-to-price coverage and fail training for unresolved held exposures.
No synthetic price fill, no survivor-only fallback and no model results until validated.
If paid access is unavailable, label the Yahoo panel incomplete and block broad
survivorship-free claims; its missing histories cannot be repaired by downloading
current ticker names again. Confirm data display and redistribution rights for SaaS.
