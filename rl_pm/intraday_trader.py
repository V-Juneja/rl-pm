from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_pm.abides_execution import load_daily_investor_source
from rl_pm.execution_train import QNetwork


def make_env(env_class, order_size):
    return env_class(
        background_config="rmsc04",
        mkt_close="09:40:00",
        timestep_duration="60s",
        starting_cash=10_000_000,
        order_fixed_size=order_size,
        state_history_length=4,
        market_data_buffer_length=5,
        first_interval="00:00:30",
        reward_mode="dense",
        done_ratio=0.3,
        debug_mode=True,
    )


def clean_state(observation, position_limit):
    state = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    state[0] /= max(1, position_limit)
    state[2:6] /= 10.0
    return np.clip(state, -10.0, 10.0)


def allowed_actions(holdings, position_limit, order_size):
    actions = [1]
    if holdings + order_size <= position_limit:
        actions.append(0)
    if holdings >= order_size:
        actions.append(2)
    return actions


def select_action(network, state, holdings, epsilon, rng, position_limit, order_size):
    valid = allowed_actions(holdings, position_limit, order_size)
    if rng.random() < epsilon:
        return rng.choice(valid)
    with torch.no_grad():
        q_values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0]
    return max(valid, key=lambda action: float(q_values[action]))


def train(env_class, episodes, seed, output_dir, position_limit, order_size):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = make_env(env_class, order_size)
    network = QNetwork(int(np.prod(env.observation_space.shape)))
    target = QNetwork(int(np.prod(env.observation_space.shape)))
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0007)
    replay = deque(maxlen=20_000)
    epsilon = 1.0
    history = []

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        state = clean_state(observation, position_limit)
        total_reward = 0.0
        actions = []
        done = False
        while not done:
            action = select_action(
                network,
                state,
                int(info["holdings"]),
                epsilon,
                rng,
                position_limit,
                order_size,
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = clean_state(next_observation, position_limit)
            done = terminated or truncated
            next_actions = allowed_actions(
                int(info["holdings"]), position_limit, order_size
            )
            replay.append(
                (state, action, float(reward), next_state, done, next_actions)
            )
            state = next_state
            total_reward += float(reward)
            actions.append(action)
            if len(replay) >= 64:
                batch = rng.sample(replay, 64)
                update_masked_network(network, target, optimizer, batch, 0.98)
            epsilon = max(0.04, epsilon * 0.997)

        history.append({"episode": episode + 1, "reward": total_reward, "epsilon": epsilon})
        if (episode + 1) % 25 == 0:
            target.load_state_dict(network.state_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output_dir / "intraday_dqn.pt")
    with (output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(history)
    env.close()
    return network, history


def update_masked_network(network, target, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones, next_actions = zip(*batch)
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    predicted = network(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        online = network(next_states)
        masked = torch.full_like(online, -1e9)
        for index, valid in enumerate(next_actions):
            masked[index, valid] = online[index, valid]
        next_choice = masked.argmax(dim=1, keepdim=True)
        next_value = target(next_states).gather(1, next_choice).squeeze(1)
        expected = rewards + gamma * next_value * (1.0 - dones)
    loss = torch.nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()


def evaluate_policy(env, policy, network, seeds, position_limit, order_size):
    rows = []
    for seed in seeds:
        observation, info = env.reset(seed=seed)
        initial_value = float(info["marked_to_market"])
        total_reward = 0.0
        actions = []
        done = False
        step = 0
        rng = random.Random(seed)
        while not done:
            holdings = int(info["holdings"])
            state = clean_state(observation, position_limit)
            if policy == "dqn":
                action = select_action(
                    network, state, holdings, 0.0, rng, position_limit, order_size
                )
            elif policy == "cash":
                action = 1
            elif policy == "buy_hold":
                action = 0 if step * order_size < position_limit else 1
            elif policy == "momentum":
                recent_return = float(state[6])
                if recent_return > 0.0 and holdings < position_limit:
                    action = 0
                elif recent_return < 0.0 and holdings >= order_size:
                    action = 2
                else:
                    action = 1
            else:
                raise ValueError(policy)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            actions.append(action)
            done = terminated or truncated
            step += 1
        final_value = float(info["marked_to_market"])
        rows.append(
            {
                "seed": seed,
                "policy": policy,
                "pnl_cents": final_value - initial_value,
                "return_bps": (final_value / initial_value - 1.0) * 10_000.0,
                "terminal_holdings": int(info["holdings"]),
                "total_reward": total_reward,
                "actions": " ".join(map(str, actions)),
            }
        )
    return rows


def summarize(rows):
    report = {}
    for policy in sorted({row["policy"] for row in rows}):
        group = [row for row in rows if row["policy"] == policy]
        pnls = [row["pnl_cents"] for row in group]
        returns = [row["return_bps"] for row in group]
        worst = min(group, key=lambda row: row["pnl_cents"])
        best = max(group, key=lambda row: row["pnl_cents"])
        report[policy] = {
            "episodes": len(group),
            "mean_pnl_cents": float(np.mean(pnls)),
            "median_pnl_cents": float(np.median(pnls)),
            "worst_episode": {
                "seed": int(worst["seed"]),
                "pnl_cents": float(worst["pnl_cents"]),
            },
            "best_episode": {
                "seed": int(best["seed"]),
                "pnl_cents": float(best["pnl_cents"]),
            },
            "positive_episode_pct": float(np.mean(np.asarray(pnls) > 0) * 100),
            "mean_return_bps": float(np.mean(returns)),
        }
    return report


def save_results(output_dir, history, rows):
    with (output_dir / "evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    rewards = np.asarray([row["reward"] for row in history])
    window = max(1, min(25, len(rewards) // 8))
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(np.arange(window, len(rewards) + 1), rolling, color="#276FBF")
    axes[0].set_title("Training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)

    names = list(report)
    means = [report[name]["mean_pnl_cents"] for name in names]
    axes[1].bar(names, means, color="#5B8E7D")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Held-out simulated P&L")
    axes[1].set_ylabel("Mean P&L (cents)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("ABIDES-NG intraday trading experiment")
    fig.tight_layout()
    fig.savefig(output_dir / "intraday_results.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a long-only intraday DQN in ABIDES-NG.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--position-limit", type=int, default=5)
    parser.add_argument("--order-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/intraday-dqn-001"))
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes, args.position_limit, args.order_size) < 1:
        parser.error("episode counts and position limit must be positive")
    if args.position_limit % args.order_size:
        parser.error("position limit must be a multiple of order size")

    env_class = load_daily_investor_source()
    network, history = train(
        env_class, args.episodes, args.seed, args.output, args.position_limit, args.order_size
    )
    env = make_env(env_class, args.order_size)
    eval_seeds = [args.seed + 1_000_000 + index for index in range(args.eval_episodes)]
    rows = []
    for policy in ("dqn", "buy_hold", "momentum", "cash"):
        rows.extend(
            evaluate_policy(
                env,
                policy,
                network,
                eval_seeds,
                args.position_limit,
                args.order_size,
            )
        )
    env.close()
    save_results(args.output, history, rows)
    metadata = {
        "algorithm": "Double DQN with long-only position mask",
        "training_episodes": len(history),
        "evaluation_episodes_per_policy": args.eval_episodes,
        "training_seed": args.seed,
        "evaluation_seed_start": eval_seeds[0],
        "evaluation_seed_end": eval_seeds[-1],
        "action": f"buy {args.order_size} shares, hold, or sell {args.order_size} shares",
        "position_limit_shares": args.position_limit,
        "market_window": "09:30-09:40 simulated time",
        "simulator": "ABIDES-NG",
        "simulator_revision": "ac7a2a7bf6e406cd2bbed887e22c5a5a790bb242",
        "price_source": "simulated",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summarize(rows), indent=2))
    print(f"Results: {args.output / 'evaluation.csv'}")
    print(f"Chart: {args.output / 'intraday_results.png'}")


if __name__ == "__main__":
    main()
