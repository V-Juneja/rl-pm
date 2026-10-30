import unittest

import numpy as np

from rl_pm.build_dataset import forward_returns, lagged_features


class DatasetTests(unittest.TestCase):
    def test_missing_quotes_are_not_zero_returns(self):
        values = forward_returns(np.array([100., 80., np.nan, 40., 20.]))
        self.assertAlmostEqual(values[0], -0.2)
        self.assertTrue(np.isnan(values[1:3]).all())
        self.assertAlmostEqual(values[3], -0.5)
        self.assertTrue(np.isnan(values[-1]))

    def test_signal_cannot_see_execution_close(self):
        prices = np.arange(100., 500.)
        expected = lagged_features(prices, prices)
        changed = prices.copy()
        changed[300:] *= 3
        actual = lagged_features(changed, prices)
        np.testing.assert_array_equal(actual[:301], expected[:301])
        self.assertFalse(np.array_equal(actual[301], expected[301]))

    def test_missing_next_quote_does_not_erase_prior_loss(self):
        values = forward_returns(np.array([[100., 20.], [10., 21.], [np.nan, 22.]]))
        self.assertAlmostEqual(values[0, 0], -0.9)
        self.assertTrue(np.isnan(values[1, 0]))
        self.assertAlmostEqual(values[1, 1], 22 / 21 - 1)


if __name__ == "__main__":
    unittest.main()
