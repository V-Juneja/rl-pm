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
from torch import nn

from rl_pm.abides_execution import clean_state, load_environment_source


class QNetwork(nn.Module):
    def __init__(self, state_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, states):
        return self.layers(states)


def build_env(env_class):
    return env_class(
        background_config="rmsc04",
        mkt_close="09:40:00",
        timestep_duration="60s",
        starting_cash=1_000_000,
        order_fixed_size=25,
        state_history_length=4,
        market_data_buffer_length=5,
        first_interval="00:00:30",
        parent_order_size=100,
        execution_window="00:05:00",
        direction="BUY",
        debug_mode=True,
    )


def choose_action(network, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(3)
    with torch.no_grad():
        values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    return int(values.argmax(dim=1).item())


def update_network(network, target, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    predicted = network(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_actions = network(next_states).argmax(dim=1, keepdim=True)
        next_values = target(next_states).gather(1, next_actions).squeeze(1)
        expected = rewards + gamma * next_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def train(env_class, episodes, seed, output_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = build_env(env_class)
    state_size = int(np.prod(env.observation_space.shape))
    network = QNetwork(state_size)
    target = QNetwork(state_size)
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0007)
    replay = deque(maxlen=20_000)
    epsilon = 1.0
    steps = 0
    losses = []
    history = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = clean_state(observation)
        total_reward = 0.0
        done = False
        while not done:
            action = choose_action(network, state, epsilon, rng)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = clean_state(next_observation)
            done = terminated or truncated
            replay.append((state, action, float(reward) / 100.0, next_state, done))
            state = next_state
            total_reward += float(reward)
            steps += 1

            if len(replay) >= 32:
                batch = rng.sample(replay, 32)
                losses.append(update_network(network, target, optimizer, batch, 0.98))
            epsilon = max(0.04, epsilon * 0.996)

        history.append({"episode": episode + 1, "reward": total_reward, "epsilon": epsilon})
        if (episode + 1) % 25 == 0:
            target.load_state_dict(network.state_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output_dir / "execution_dqn.pt")
    with (output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(history)
    env.close()
    return network, history, losses


def policy_action(policy, network, observation, step, rng):
    if policy == "dqn":
        return choose_action(network, clean_state(observation), 0.0, rng)
    if policy == "market_twap":
        return 0
    if policy == "passive_then_market":
        return 0 if step >= 3 else 1
    if policy == "limit_first_then_market":
        return 1 if step == 0 else 0
    if policy == "limit_only":
        return 1
    raise ValueError(policy)


def run_episode(env, policy, network, seed):
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_cost = 0.0
    actions = []
    done = False
    step = 0
    final_info = {}
    rng = random.Random(seed)
    while not done:
        action = policy_action(policy, network, observation, step, rng)
        actions.append(action)
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        total_cost += float(final_info.get("pnl", 0.0))
        done = terminated or truncated
        step += 1

    filled = int(final_info.get("holdings", 0))
    entry_price = float(env.entry_price)
    shortfall_bps = (
        -total_cost / (entry_price * filled) * 10_000.0 if filled > 0 else None
    )
    return {
        "seed": seed,
        "policy": policy,
        "reward": total_reward,
        "filled_shares": filled,
        "overfill_shares": max(0, filled - env.parent_order_size),
        "completion_pct": 100.0 * min(filled, env.parent_order_size) / env.parent_order_size,
        "shortfall_bps": shortfall_bps,
        "actions": " ".join(map(str, actions)),
    }


def evaluate(env_class, network, seeds):
    env = build_env(env_class)
    rows = []
    for policy in (
        "dqn",
        "market_twap",
        "limit_first_then_market",
        "passive_then_market",
        "limit_only",
    ):
        for seed in seeds:
            rows.append(run_episode(env, policy, network, seed))
    env.close()
    return rows


def summarize(rows):
    report = {}
    for policy in sorted({row["policy"] for row in rows}):
        group = [row for row in rows if row["policy"] == policy]
        shortfalls = [row["shortfall_bps"] for row in group if row["shortfall_bps"] is not None]
        report[policy] = {
            "episodes": len(group),
            "mean_reward": float(np.mean([row["reward"] for row in group])),
            "mean_completion_pct": float(np.mean([row["completion_pct"] for row in group])),
            "full_completion_pct": float(np.mean([row["completion_pct"] >= 100.0 for row in group]) * 100),
            "mean_overfill_shares": float(np.mean([row["overfill_shares"] for row in group])),
            "mean_shortfall_bps_when_filled": float(np.mean(shortfalls)) if shortfalls else None,
        }
    return report


def save_results(output_dir, history, losses, rows):
    with (output_dir / "evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rewards = np.asarray([row["reward"] for row in history])
    window = max(1, min(25, len(rewards) // 8))
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(np.arange(window, len(rewards) + 1), rolling, color="#276FBF", linewidth=1.8)
    axes[0].set_title("Training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].grid(alpha=0.25)

    policies = list(report)
    completion = [report[name]["mean_completion_pct"] for name in policies]
    shortfall = [report[name]["mean_shortfall_bps_when_filled"] for name in policies]
    positions = np.arange(len(policies))
    axes[1].bar(positions, completion, color="#5B8E7D")
    axes[1].set_xticks(positions, policies, rotation=18, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Mean order completion (%)")
    axes[1].set_title("Held-out simulator seeds")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(shortfall):
        if value is not None:
            axes[1].text(index, min(99, completion[index]) - 11, f"{value:.1f} bps", ha="center", fontsize=8)
    fig.suptitle("ABIDES-NG execution policy experiment")
    fig.tight_layout()
    fig.savefig(output_dir / "execution_results.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a DQN order-execution policy in ABIDES-NG.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", type=Path, default=Path("outputs/execution-dqn-001"))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("episode counts must be positive")

    env_class = load_environment_source()
    if args.eval_only:
        checkpoint = args.output / "execution_dqn.pt"
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
        observation_size = int(np.prod(build_env(env_class).observation_space.shape))
        network = QNetwork(observation_size)
        network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with (args.output / "training.csv").open(newline="") as file:
            history = list(csv.DictReader(file))
        history = [{"episode": int(row["episode"]), "reward": float(row["reward"]), "epsilon": float(row["epsilon"])} for row in history]
        losses = []
    else:
        network, history, losses = train(env_class, args.episodes, args.seed, args.output)
    eval_seeds = [args.seed + 1_000_000 + index for index in range(args.eval_episodes)]
    rows = evaluate(env_class, network, eval_seeds)
    save_results(args.output, history, losses, rows)
    metadata = {
        "algorithm": "Double DQN",
        "training_episodes": len(history),
        "evaluation_episodes_per_policy": args.eval_episodes,
        "training_seed": args.seed,
        "evaluation_seed_start": eval_seeds[0],
        "evaluation_seed_end": eval_seeds[-1],
        "parent_order_shares": 100,
        "child_order_shares": 25,
        "execution_window": "5 minutes",
        "simulator": "ABIDES-NG",
        "simulator_revision": "ac7a2a7bf6e406cd2bbed887e22c5a5a790bb242",
        "price_source": "simulated",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summarize(rows), indent=2))
    print(f"Training history: {args.output / 'training.csv'}")
    print(f"Evaluation results: {args.output / 'evaluation.csv'}")
    print(f"Chart: {args.output / 'execution_results.png'}")


if __name__ == "__main__":
    main()
