# Feature inputs and historical-data recovery

Finished the feature module and improved historical-data recovery. The new dated-security snapshot has 7.22M price rows across 1,228 securities, with 99.995% membership-day price availability. All 31 tests pass. Next: validate terminal payouts and prepare the training dataset.

The figures are generated from the local saved dataset and feature module:

```sh
.venv/bin/python scripts/create_journal_images.py
```

- `outputs/journal/01-data-coverage.png`
- `outputs/journal/02-training-features.png`
