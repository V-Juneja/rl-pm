from datetime import date
import unittest

from rl_pm.extend_histories import aligned_extension
from rl_pm.recover_terminals import match_event


class RecoveryTests(unittest.TestCase):
    def test_reused_ticker_requires_a_nearby_historical_event(self):
        gap = {"security_id": "AGN-201503", "execution_date": "2015-03-16"}
        old = {"ticker": "AGN", "last_trade_date": "2015-03-16", "sec_id": "old"}
        new = {"ticker": "AGN", "last_trade_date": "2020-05-08", "sec_id": "new"}
        self.assertEqual(match_event(gap, [old, new])["sec_id"], "old")
        self.assertIsNone(match_event(gap, [new]))
        self.assertIsNone(match_event(gap, [old, dict(old, sec_id="ambiguous")]))

    def test_extension_stops_before_legal_cancellation(self):
        days = [date(2009, 9, n) for n in range(1, 26)]
        reference = dict(zip(days[:20], range(100, 140, 2)))
        candidate = dict(zip(days, range(50, 75)))
        extra, scale, overlap, deviation = aligned_extension(reference, candidate, days[23])
        self.assertEqual(overlap, 20)
        self.assertEqual(scale, 2)
        self.assertEqual(deviation, 0)
        self.assertEqual(set(extra), set(days[20:23]))
        self.assertEqual(extra[days[20]], 140)

    def test_wrong_identity_or_price_basis_cannot_be_spliced(self):
        days = [date(2009, 9, n) for n in range(1, 26)]
        reference = dict(zip(days[:20], range(100, 120)))
        candidate = dict(zip(days, range(100, 125)))
        candidate[days[15]] *= 3
        with self.assertRaises(ValueError):
            aligned_extension(reference, candidate, days[-1])


if __name__ == "__main__":
    unittest.main()
