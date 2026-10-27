import csv
from datetime import date
from pathlib import Path
import tempfile
import unittest
from rl_pm.coverage import audit, valid_dates


class CoverageTests(unittest.TestCase):
    def test_membership_changes_and_missing_security(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'membership.csv').write_text('date,tickers\n2020-01-02,OLD\n2020-01-03,NEW\n')
            (root/'SPY_yahoo_daily.csv').write_text('date,adjusted_close\n2020-01-02,100\n2020-01-03,101\n')
            (root/'OLD_yahoo_daily.csv').write_text('date,adjusted_close\n2020-01-02,10\n')
            result = audit(root, date(2020, 1, 2), date(2020, 1, 3))
            self.assertEqual(result['required_asset_sessions'], 2)
            self.assertEqual(result['missing_asset_sessions'], 1)
            self.assertEqual(result['absent_files'], 1)

    def test_bad_adjusted_prices_are_not_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prices.csv'
            path.write_text('date,adjusted_close\n2020-01-02,nan\n2020-01-03,0\n2020-01-06,5\n')
            days, invalid = valid_dates(path)
            self.assertEqual(days, {date(2020, 1, 6)})
            self.assertEqual(invalid, 2)
