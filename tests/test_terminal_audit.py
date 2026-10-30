import unittest

import numpy as np

from rl_pm.terminal_audit import apply_suspensions, cash_entitlement_return


class TerminalTests(unittest.TestCase):
    def test_suspension_excludes_quotes_but_preserves_resumption_loss(self):
        dates = np.array(["2023-03-09", "2023-03-10", "2023-03-21", "2023-03-28"], dtype="datetime64[D]")
        prices = np.array([[106.04], [106.04], [106.04], [0.40]])
        intervals = [{"security_id": "SIVB", "last_trade": dates[0], "next_trade": dates[-1]}]
        count = apply_suspensions(prices, dates, np.array(["SIVB"]), intervals)
        self.assertEqual(count, 2)
        self.assertTrue(np.isnan(prices[1:3]).all())
        self.assertAlmostEqual(prices[-1, 0] / prices[0, 0] - 1, -0.99622783855)

    def test_entitlement_uses_nominal_not_adjusted_price(self):
        self.assertAlmostEqual(cash_entitlement_return(95, 94), 95 / 94 - 1)
        self.assertEqual(cash_entitlement_return(0, 94), -1)
        with self.assertRaises(ValueError):
            cash_entitlement_return(95, 0)


if __name__ == "__main__":
    unittest.main()
