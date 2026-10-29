from datetime import datetime
import unittest

from rl_pm.curated_prices import base_ticker, identity_segments


class CuratedPricesTests(unittest.TestCase):
    def test_symbol_reuse_is_mapped_by_date(self):
        intervals = [
            ("ABC-200101", datetime(1996, 1, 1), datetime(2001, 1, 1)),
            ("ABC", datetime(2010, 1, 1), datetime(2020, 1, 1)),
        ]
        rules, ambiguous = identity_segments(intervals, {"ABC"})
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["price_ticker"], "ABC-200101")
        self.assertEqual(rules[0]["through"], "2000-12-31")
        self.assertEqual(ambiguous, [])

    def test_two_simultaneous_securities_are_not_guessed(self):
        intervals = [
            ("ABC-200101", datetime(1996, 1, 1), datetime(2001, 1, 1)),
            ("ABC-200901", datetime(1999, 1, 1), datetime(2009, 1, 1)),
        ]
        rules, ambiguous = identity_segments(intervals, {"ABC"})
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(ambiguous[0]["from"], "1999-01-01")
        self.assertEqual(ambiguous[0]["through"], "2000-12-31")
        self.assertEqual(len(rules), 2)

    def test_only_dated_security_suffixes_are_removed(self):
        self.assertEqual(base_ticker("SUN-201210"), "SUN")
        self.assertEqual(base_ticker("BRK-B"), "BRK.B")
        self.assertEqual(base_ticker("AZA-A-200106"), "AZA.A")


if __name__ == "__main__":
    unittest.main()
