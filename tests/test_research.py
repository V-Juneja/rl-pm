import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rl_pm.allocation_policy import CashAndStocks
from rl_pm.research_data import valuation_returns
from rl_pm.research_environment import ResearchPortfolio
from rl_pm.train import Normalizer


class ResearchTests(unittest.TestCase):
    def test_gap_defers_real_loss(self):
        np.testing.assert_allclose(valuation_returns([100., np.nan, np.nan, .4]), [0, 0, -.996])

    def test_cash_weight_does_not_shrink_with_universe(self):
        for count in (10, 500):
            distribution = CashAndStocks(torch.zeros(count), torch.zeros(2))
            self.assertAlmostEqual(float(distribution.mean[-1]), .5)
            action = distribution.sample()
            self.assertTrue(torch.isfinite(distribution.log_prob(action)))
            self.assertAlmostEqual(float(action.sum()), 1, places=5)

    def test_suspended_position_cannot_be_sold(self):
        market = SimpleNamespace(
            ids=np.array(['BANK']), dates=np.array(['2023-03-10'], dtype='datetime64[D]'),
            features=np.zeros((1, 1, 1), dtype=np.float32), eligible=np.ones((1, 1), dtype=bool),
            tradable=np.zeros((1, 1), dtype=bool), returns=np.array([[-.996]]),
            cash_returns=np.zeros(1), benchmark=np.zeros(1),
            cash_settlements=np.zeros((1, 1), dtype=bool), assumed_exits=np.zeros((1, 1), dtype=bool),
        )
        env = ResearchPortfolio(market, Normalizer([0], [1]), 0, 1, 10)
        env.weights[:] = .5
        env.cash = .5
        observation = env.observation()
        self.assertEqual(len(observation['indices']), 0)
        _, _, info = env.step([], [1.])
        self.assertAlmostEqual(info['return'], -.498)
        self.assertAlmostEqual(info['turnover'], 0)


if __name__ == '__main__':
    unittest.main()
