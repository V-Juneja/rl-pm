# rl-pm

An experimental RL portfolio research engine for selectively holding, excluding,
and re-entering historical S&P 500 constituents.

## Current status
Initial accounting engine, candidate membership audit, and a synthetic tabular
Q-learning smoke experiment. No real-market model has been trained. Historical
prices and delisting coverage are still required; no survivorship-free claim is made.

## Run
Python 3.10+; standard library only for the initial prototype.

```sh
python3 -m unittest discover -s tests -v
python3 -m rl_pm.demo
python3 scripts/fetch_membership.py
python3 -m rl_pm.audit
```

The demo uses generated returns, not market data. Its results cannot demonstrate
investment performance. The market-data download is exploratory membership only.
See [research plan](docs/RESEARCH_PLAN.md) for data sources, limitations and roadmap.
