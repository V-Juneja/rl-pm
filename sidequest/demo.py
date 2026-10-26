"""Small tabular Q-learning smoke experiment on SYNTHETIC data, not stock evidence.

State: previous return signs. Actions: cash, A, B, equal weight.
Chronological train/test split; Q-values are frozen during evaluation.
"""
import json
import random
from .engine import rebalance

ACTIONS = [{}, {'A': 1}, {'B': 1}, {'A': .5, 'B': .5}]


def run(seed=7, episodes=80):
    rng = random.Random(seed)
    rows = [{'A': rng.gauss(.0002, .012), 'B': rng.gauss(.0001, .01)}
            for _ in range(1600)]
    def state(row):
        return tuple(int(row[s] > 0) for s in ('A', 'B'))
    q = {s: [0.] * 4 for s in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    for _ in range(episodes):
        weights = {}
        for t in range(1, 1100):
            s = state(rows[t - 1])
            a = rng.randrange(4) if rng.random() < .15 else max(range(4), key=lambda a: q[s][a])
            reward, weights, _ = rebalance(weights, ACTIONS[a], rows[t])
            q[s][a] += .05 * (reward + .9 * max(q[state(rows[t])]) - q[s][a])
    results = {}
    for policy in ('q_learning', 'equal_weight', 'cash'):
        equity, peak, max_dd, total_turnover, weights = 1., 1., 0., 0., {}
        for t in range(1100, len(rows)):
            a = max(range(4), key=lambda a: q[state(rows[t-1])][a]) if policy == 'q_learning' else (3 if policy == 'equal_weight' else 0)
            ret, weights, turnover = rebalance(weights, ACTIONS[a], rows[t])
            equity *= 1 + ret
            peak = max(peak, equity)
            max_dd = max(max_dd, 1 - equity / peak)
            total_turnover += turnover
        results[policy] = dict(total_return=equity-1, max_drawdown=max_dd, turnover=total_turnover)
    return dict(dataset='SYNTHETIC — NOT MARKET PERFORMANCE', seed=seed,
                limitation='Shared policy omits current holdings from state; pipeline smoke test only.',
                train_days=1099, test_days=500, results=results)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
