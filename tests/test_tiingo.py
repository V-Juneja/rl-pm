import tempfile
import unittest
from pathlib import Path

from rl_pm.tiingo import membership_starts, missing_windows, parse_prices


class TiingoTests(unittest.TestCase):
    def test_history_starts_at_first_membership_even_with_unsorted_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            path.write_text('date,tickers\n2005-01-03,"ABC,XYZ"\n1996-01-02,ABC\n')
            starts = membership_starts(path)
        self.assertEqual(str(starts["ABC"]), "1996-01-02")
        self.assertEqual(str(starts["XYZ"]), "2005-01-03")

    def test_groups_incomplete_windows_and_skips_complete_events(self):
        contents = (
            "ticker,price_status,price_from,price_through\n"
            "ABC,missing_file,2001-01-02,2001-01-31\n"
            "ABC,post_removal_gaps,2000-06-01,2000-06-30\n"
            "XYZ,price_window_complete,2002-01-02,2002-01-31\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            path.write_text(contents)
            windows = missing_windows(path)
        self.assertEqual(str(windows["ABC"]["start"]), "2000-06-01")
        self.assertEqual(str(windows["ABC"]["end"]), "2001-01-31")
        self.assertNotIn("XYZ", windows)

    def test_maps_tiingo_adjusted_close_to_project_schema(self):
        rows = parse_prices([{
            "date": "2001-01-02T00:00:00.000Z", "open": 10, "high": 12,
            "low": 9, "close": 11, "volume": 1000, "adjClose": 10.5,
        }])
        self.assertEqual(rows[0]["date"], "2001-01-02")
        self.assertEqual(rows[0]["adjusted_close"], 10.5)

    def test_discards_invalid_adjusted_close(self):
        rows = parse_prices([{
            "date": "2001-01-02", "open": 10, "high": 12,
            "low": 9, "close": 11, "volume": 1000, "adjClose": 0,
        }])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
