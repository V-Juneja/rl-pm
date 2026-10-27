# rl-pm

An experimental RL portfolio research engine for selectively holding, excluding,
and re-entering historical S&P 500 constituents.

## Current status
Initial accounting engine and candidate membership audit.
No RL model is currently implemented or trained. Historical
prices and delisting coverage are still required; no survivorship-free claim is made.

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
