import unittest
from rl_pm.engine import rebalance, equal_weight


class AccountingTests(unittest.TestCase):
    def test_terminal_loss_is_not_dropped(self):
        ret, weights, _ = rebalance({}, {'DEAD': .5}, {'DEAD': -1}, 0)
        self.assertEqual(ret, -.5)
        self.assertEqual(weights['DEAD'], 0)

    def test_missing_held_return_fails(self):
        with self.assertRaises(ValueError):
            rebalance({}, {'DEAD': 1}, {})

    def test_exit_cost_and_cash(self):
        ret, weights, turn = rebalance({'A': 1}, {}, {}, 10)
        self.assertAlmostEqual(ret, -.001)
        self.assertEqual(weights, {})
        self.assertEqual(turn, 1)

    def test_drift_and_cost(self):
        ret, weights, _ = rebalance({}, {'A': .5}, {'A': .1}, 10)
        self.assertAlmostEqual(ret, .9995 * 1.05 - 1)
        self.assertAlmostEqual(weights['A'], .55 / 1.05)

    def test_no_leverage(self):
        with self.assertRaises(ValueError):
            rebalance({}, {'A': 2}, {'A': 0})

    def test_empty_universe_is_cash(self):
        self.assertEqual(equal_weight([]), {})
