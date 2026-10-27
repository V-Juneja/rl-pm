import unittest
from rl_pm.recovery import assess, candidates, parse_chart


class RecoveryTests(unittest.TestCase):
    def test_symbol_variants_are_candidates(self):
        self.assertEqual(candidates('BRK.B'), ['BRK.B', 'BRK-B'])
        self.assertEqual(candidates('ABC'), ['ABC'])

    def test_current_reused_symbol_fails_historical_date_match(self):
        self.assertEqual(assess([{'date': '2025-01-02'}],
                               [{'pre_removal_session': '2000-01-03'}]), 0)

    def test_invalid_adjusted_prices_are_not_retained(self):
        payload = {'chart': {'result': [{
            'timestamp': [946857600, 946944000],
            'indicators': {'quote': [{}], 'adjclose': [{'adjclose': [None, 12]}]},
        }]}}
        rows, _ = parse_chart(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['adjusted_close'], 12)

    def test_no_fabricated_adjusted_prices(self):
        with self.assertRaises(ValueError):
            parse_chart({'chart': {'result': [{
                'timestamp': [946857600], 'indicators': {'quote': [{'close': [12]}]},
            }]}})
