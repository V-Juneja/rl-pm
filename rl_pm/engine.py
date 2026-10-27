"""Long-only daily accounting. Actions execute at a supplied execution boundary.

Returns MUST cover the interval after that boundary, including terminal payouts.
No missing-return imputation is permitted for held assets.
"""
import math


def rebalance(weights, target, returns, cost_bps=10):
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('Invalid transaction cost')
    for allocation in (weights, target):
        if any(not math.isfinite(v) or v < 0 for v in allocation.values()):
            raise ValueError('Weights must be finite and long-only')
        if sum(allocation.values()) > 1 + 1e-10:
            raise ValueError('Leverage is disabled')
    turnover = sum(abs(target.get(s, 0) - weights.get(s, 0))
                   for s in weights.keys() | target.keys())
    cost = turnover * cost_bps / 10000
    growth = 1 - sum(target.values())
    values = {}
    for symbol, weight in target.items():
        if not weight:
            continue
        if symbol not in returns:
            raise ValueError(f'Missing held-asset return: {symbol}')
        ret = returns[symbol]
        if not math.isfinite(ret) or ret < -1:
            raise ValueError(f'Invalid return: {symbol}')
        values[symbol] = weight * (1 + ret)
        growth += values[symbol]
    # Costs reduce capital before exposure; cash is implicit in returned weights.
    net_growth = (1 - cost) * growth
    if net_growth <= 0:
        raise ValueError('Portfolio insolvent')
    drifted = {s: v / growth for s, v in values.items()}
    return net_growth - 1, drifted, turnover


def equal_weight(members):
    members = sorted(set(members))
    return {s: 1 / len(members) for s in members}
