import csv
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_pm import data_readiness as audit


class ReadinessTests(unittest.TestCase):
    def test_predecessor_mapping_does_not_reuse_successor_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "NEW_combined_daily.csv").write_text(
                "date,adjusted_close\n2000-01-03,999\n2001-01-03,100\n"
            )
            (folder / "OLD_combined_daily.csv").write_text(
                "date,adjusted_close\n2000-01-04,10\n2001-01-04,11\n"
            )
            rules = [{"membership_ticker": "NEW", "price_ticker": "OLD",
                      "from": "1990-01-01", "through": "2000-12-31"}]
            with patch.object(audit, "COMBINED", folder):
                dates, invalid, duplicates = audit.membership_price_dates("NEW", rules)
            self.assertEqual(dates, {date(2000, 1, 4), date(2001, 1, 3)})

    def test_nonfinite_prices_do_not_count_as_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("date,adjusted_close\n2000-01-03,nan\n2000-01-04,inf\n")
            dates, invalid, duplicates = audit.read_price_dates(path)
            self.assertEqual(dates, set())
            self.assertEqual(invalid, 2)


if __name__ == "__main__":
    unittest.main()
