import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet

from .cross_sectional import (
    RANK_FEATURES,
    augment_features,
    extra_feature_count,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 7
    updates: int = 300
    episode_steps: int = 252
    hidden: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch_size: int = 32
    value_weight: float = 0.5
    max_gradient: float = 0.5
    target_kl: float = 0.03
    cost_bps: float = 10.0
    reward_scale: float = 100.0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


class Market:
    def __init__(self, path):
        self.path = path
        self.digest = file_hash(path)

        metadata_path = path.with_suffix(".json")
        self.metadata = json.loads(metadata_path.read_text())

        checks = (
            "identifiers_verified",
            "membership_verified",
            "execution_alignment_verified",
            "terminal_returns_verified",
            "features_point_in_time",
        )

        if self.metadata.get("sha256") != self.digest:
            raise ValueError("Dataset hash does not match its audit")

        failed = [
            check
            for check in checks
            if self.metadata.get(check) is not True
        ]

        if failed:
            raise ValueError(
                "Dataset checks incomplete: " + ", ".join(failed)
            )

        with np.load(path, allow_pickle=False) as source:
            self.dates = source["dates"].astype("datetime64[D]")
            self.ids = source["security_ids"].astype(str)
            self.features = source["features"].astype(np.float32)
            self.eligible = source["eligible"].astype(bool)
            self.tradable = source["tradable"].astype(bool)
            self.returns = source["returns"].astype(np.float64)
            self.benchmark = source["benchmark"].astype(np.float64)
            self.cash_returns = source["cash_returns"].astype(
                np.float64
            )

        self.validate()

    def validate(self):
        if self.features.ndim != 3:
            raise ValueError("features must have shape T,N,F")

        periods, securities, feature_count = self.features.shape

        if min(periods, securities, feature_count) < 1:
            raise ValueError("Dataset dimensions cannot be empty")

        if self.dates.shape != (periods,):
            raise ValueError("Date shape mismatch")

        if self.ids.shape != (securities,):
            raise ValueError("Security ID shape mismatch")

        if len(set(self.ids)) != securities:
            raise ValueError("Security IDs must be unique")

        if np.isnat(self.dates).any():
            raise ValueError("Invalid dates")

        if not np.all(self.dates[1:] > self.dates[:-1]):
            raise ValueError("Dates must be unique and increasing")

        for name in ("eligible", "tradable", "returns"):
            if getattr(self, name).shape != (periods, securities):
                raise ValueError(f"{name} shape mismatch")

        for name in ("benchmark", "cash_returns"):
            values = getattr(self, name)

            if values.shape != (periods,):
                raise ValueError(f"{name} shape mismatch")

            if not np.isfinite(values).all() or (values <= -1).any():
                raise ValueError(f"Invalid {name}")

        finite_returns = self.returns[np.isfinite(self.returns)]

        if (finite_returns < -1).any():
            raise ValueError("Stock returns cannot be below -100%")

        if np.isinf(self.returns).any():
            raise ValueError("Infinite stock returns")

        required = self.eligible & self.tradable

        if not np.isfinite(self.returns[required]).all():
            raise ValueError("Missing return for buyable security")

        if not np.isfinite(self.features[self.eligible]).all():
            raise ValueError("Missing eligible-security features")

    def bounds(self, start, end):
        left = int(np.searchsorted(
            self.dates,
            np.datetime64(start),
            side="left",
        ))
        right = int(np.searchsorted(
            self.dates,
            np.datetime64(end),
            side="left",
        ))

        if right - left < 2:
            raise ValueError(f"Insufficient observations: {start}–{end}")

        return left, right


class Normalizer:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, market, bounds):
        left, right = bounds
        values = market.features[left:right]
        mask = market.eligible[left:right]
        sample = values[mask]

        if len(sample) < 2:
            raise ValueError("Insufficient training features")

        mean = sample.mean(axis=0)
        scale = np.maximum(sample.std(axis=0), 0.000001)
        added = extra_feature_count(values.shape[-1])
        if added:
            mean = np.concatenate([mean, np.full(added, 0.5)])
            scale = np.concatenate([
                scale,
                np.full(added, 1 / np.sqrt(12)),
            ])
        return cls(mean, scale)

    def transform(self, values):
        return np.clip(
            (values - self.mean) / self.scale,
            -10,
            10,
        ).astype(np.float32)


class Portfolio:
    def __init__(self, market, normalizer, left, right, cost_bps):
        self.market = market
        self.normalizer = normalizer
        self.left = left
        self.right = right
        self.cost_rate = cost_bps / 10000
        self.reset()

    def reset(self):
        self.position = self.left
        self.weights = np.zeros(
            len(self.market.ids),
            dtype=np.float64,
        )
        self.cash = 1.0
        self.equity = 1.0

    def observation(self):
        t = self.position
        active = self.market.eligible[t]
        held = self.weights > 0

        if np.any(held & ~self.market.tradable[t]):
            raise ValueError(
                f"Held security cannot execute on "
                f"{self.market.dates[t]}"
            )

        relevant = active | held
        values = augment_features(
            self.market.features[t], active, held,
        ).copy()

        if not np.isfinite(values[relevant]).all():
            raise ValueError(
                f"Missing held/eligible features on "
                f"{self.market.dates[t]}"
            )

        values[~relevant] = self.normalizer.mean

        features = self.normalizer.transform(values)
        state = np.column_stack([
            features,
            self.weights.astype(np.float32),
            active.astype(np.float32),
            self.market.tradable[t].astype(np.float32),
        ])

        buyable = active & self.market.tradable[t]
        indices = np.flatnonzero(buyable)

        return {
            "state": torch.from_numpy(state),
            "indices": torch.from_numpy(indices),
            "cash": torch.tensor(self.cash, dtype=torch.float32),
        }

    def step(self, indices, action):
        t = self.position
        indices = np.asarray(indices, dtype=np.int64)
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (len(indices) + 1,):
            raise ValueError("Allocation shape mismatch")

        if not np.isfinite(action).all() or (action < 0).any():
            raise ValueError("Invalid allocation")

        if not np.isclose(action.sum(), 1, atol=0.00001):
            raise ValueError("Allocations must sum to one")

        target = np.zeros_like(self.weights)
        target[indices] = action[:-1]
        cash_target = float(action[-1])

        used = target > 0
        returns = self.market.returns[t]

        if not np.isfinite(returns[used]).all():
            raise ValueError(
                f"Missing held return on {self.market.dates[t]}"
            )

        low, high = 0.0, self.cost_rate * 2

        for _ in range(40):
            fee = (low + high) / 2
            traded = np.abs(
                (1 - fee) * target - self.weights
            ).sum()

            if fee > self.cost_rate * traded:
                high = fee
            else:
                low = fee

        fee = (low + high) / 2
        turnover = np.abs(
            (1 - fee) * target - self.weights
        ).sum()

        stock_values = np.zeros_like(target)
        stock_values[used] = (
            target[used] * (1 + returns[used])
        )

        cash_value = cash_target * (
            1 + self.market.cash_returns[t]
        )
        gross_growth = stock_values.sum() + cash_value
        net_growth = (1 - fee) * gross_growth

        if not np.isfinite(net_growth) or net_growth < 0:
            raise ValueError("Invalid portfolio growth")

        portfolio_return = net_growth - 1
        self.equity *= net_growth

        if gross_growth > 0:
            self.weights = stock_values / gross_growth
            self.cash = cash_value / gross_growth
        else:
            self.weights[:] = 0
            self.cash = 0

        self.position += 1
        bankrupt = net_growth <= 0
        done = bankrupt or self.position >= self.right

        info = {
            "date": str(self.market.dates[t]),
            "return": float(portfolio_return),
            "benchmark_return": float(self.market.benchmark[t]),
            "turnover": float(turnover),
            "cost_fraction": float(fee),
            "equity": float(self.equity),
            "cash_weight": float(self.cash),
            "bankrupt": bankrupt,
        }

        reward = float(np.log(max(net_growth, 1e-12)))
        return reward, done, info


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.stock_head = nn.Linear(hidden, 1)
        self.cash_head = nn.Linear(hidden + 1, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observation):
        state = observation["state"]
        indices = observation["indices"]
        encoded = self.encoder(state)

        eligible = state[:, -2] > 0
        held = state[:, -3] > 0
        relevant = eligible | held

        if relevant.any():
            pooled = encoded[relevant].mean(dim=0)
        else:
            pooled = encoded.new_zeros(encoded.shape[-1])

        context = torch.cat([
            pooled,
            observation["cash"].reshape(1),
        ])

        stock_scores = self.stock_head(
            encoded[indices]
        ).squeeze(-1)
        cash_score = self.cash_head(context).reshape(1)

        scores = torch.cat([stock_scores, cash_score])
        concentration = torch.nn.functional.softplus(scores) + 0.2
        value = self.value_head(context).squeeze()

        if len(indices) == 0:
            return None, value

        return Dirichlet(concentration), value

    def act(self, observation, deterministic=False):
        distribution, value = self(observation)

        if distribution is None:
            action = observation["state"].new_ones(1)
            log_probability = value.new_zeros(())
        else:
            action = (
                distribution.mean
                if deterministic
                else distribution.sample()
            )
            log_probability = distribution.log_prob(action)

        return action, log_probability, value

    def score(self, observation, action):
        distribution, value = self(observation)

        if distribution is None:
            return value.new_zeros(()), value

        return distribution.log_prob(action), value


def collect(model, environment, config):
    records = []

    while True:
        observation = environment.observation()

        with torch.no_grad():
            action, log_probability, value = model.act(observation)

        reward, done, info = environment.step(
            observation["indices"].numpy(),
            action.numpy(),
        )

        records.append({
            "observation": observation,
            "action": action,
            "log_probability": float(log_probability),
            "value": float(value),
            "reward": reward * config.reward_scale,
            "done": done,
        })

        if done:
            break

    advantage = 0.0
    next_value = 0.0

    for record in reversed(records):
        continuation = 0.0 if record["done"] else 1.0
        delta = (
            record["reward"]
            + config.gamma * next_value * continuation
            - record["value"]
        )
        advantage = (
            delta
            + config.gamma
            * config.gae_lambda
            * continuation
            * advantage
        )
        record["advantage"] = advantage
        record["target"] = advantage + record["value"]
        next_value = record["value"]

    advantages = np.array([
        record["advantage"] for record in records
    ])
    mean = advantages.mean()
    scale = max(advantages.std(), 1e-8)

    for record in records:
        record["advantage"] = (
            record["advantage"] - mean
        ) / scale

    return records


def optimize(model, optimizer, records, config):
    statistics = []

    for _ in range(config.epochs):
        order = np.random.permutation(len(records))

        for offset in range(0, len(order), config.batch_size):
            batch = [
                records[index]
                for index in order[offset:offset + config.batch_size]
            ]

            probabilities = []
            values = []

            for record in batch:
                probability, value = model.score(
                    record["observation"],
                    record["action"],
                )
                probabilities.append(probability)
                values.append(value)

            new_log = torch.stack(probabilities)
            predictions = torch.stack(values)
            old_log = torch.tensor([
                record["log_probability"] for record in batch
            ], dtype=torch.float32)
            advantage = torch.tensor([
                record["advantage"] for record in batch
            ], dtype=torch.float32)
            targets = torch.tensor([
                record["target"] for record in batch
            ], dtype=torch.float32)

            log_ratio = new_log - old_log
            ratio = torch.exp(log_ratio.clamp(-20, 20))
            clipped = ratio.clamp(
                1 - config.clip,
                1 + config.clip,
            )

            policy_loss = -torch.minimum(
                ratio * advantage,
                clipped * advantage,
            ).mean()
            value_loss = torch.nn.functional.mse_loss(
                predictions,
                targets,
            )
            loss = (
                policy_loss
                + config.value_weight * value_loss
            )

            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_gradient,
            )
            optimizer.step()

            with torch.no_grad():
                approximate_kl = (
                    ratio - 1 - log_ratio
                ).mean().item()

            statistics.append({
                "loss": float(loss.detach()),
                "kl": approximate_kl,
            })

            if approximate_kl > config.target_kl:
                return statistics

    return statistics


def performance(rows):
    returns = np.array([row["return"] for row in rows])
    equity = np.cumprod(1 + returns)
    path = np.concatenate([[1.0], equity])
    peaks = np.maximum.accumulate(path)
    drawdown = 1 - path / peaks
    deviation = returns.std(ddof=1) if len(returns) > 1 else 0

    return {
        "total_return": float(equity[-1] - 1),
        "max_drawdown": float(drawdown.max()),
        "annualized_volatility": float(
            deviation * np.sqrt(252)
        ),
        "return_volatility_ratio": float(
            returns.mean() / deviation * np.sqrt(252)
        ) if deviation > 0 else None,
        "turnover": float(sum(row["turnover"] for row in rows)),
        "intervals": len(rows),
    }


def evaluate(
    model,
    market,
    normalizer,
    bounds,
    config,
    policy,
    collect_attribution=False,
):
    environment = getattr(market, "environment_class", Portfolio)(
        market,
        normalizer,
        *bounds,
        config.cost_bps,
    )
    environment.capture_attribution = collect_attribution
    rows = []
    attribution = {}

    while True:
        observation = environment.observation()
        indices = observation["indices"].numpy()

        if policy == "agent":
            with torch.no_grad():
                action, _, _ = model.act(
                    observation,
                    deterministic=True,
                )
            allocation = action.numpy()
        elif policy == "equal_weight" and len(indices):
            active = market.eligible[environment.position, indices].astype(float)
            if active.sum():
                allocation = np.concatenate([active / active.sum(), [0.0]])
            else:
                allocation = np.concatenate([active, [1.0]])
        else:
            allocation = np.zeros(len(indices) + 1)
            allocation[-1] = 1

        _, done, info = environment.step(indices, allocation)
        for item in info.pop("attribution", []):
            security_id = item["security_id"]
            row = attribution.setdefault(security_id, {
                "security_id": security_id,
                "sessions_held": 0,
                "exposure_days": 0.0,
                "capital_contribution": 0.0,
                "best_daily_contribution": 0.0,
                "best_date": "",
                "worst_daily_contribution": 0.0,
                "worst_date": "",
                "assumed_terminal_sessions": 0,
                "assumed_terminal_contribution": 0.0,
            })
            contribution = float(item["capital_contribution"])
            weight = float(item["weight"])
            row["sessions_held"] += int(
                weight > 0
                and security_id not in {"CASH", "TRADING_COSTS"}
            )
            row["exposure_days"] += weight
            row["capital_contribution"] += contribution
            if contribution > row["best_daily_contribution"]:
                row["best_daily_contribution"] = contribution
                row["best_date"] = item["date"]
            if contribution < row["worst_daily_contribution"]:
                row["worst_daily_contribution"] = contribution
                row["worst_date"] = item["date"]
            if item["assumed_terminal"]:
                row["assumed_terminal_sessions"] += 1
                row["assumed_terminal_contribution"] += contribution
        rows.append(info)

        if done:
            break

    return (rows, list(attribution.values())) if collect_attribution else rows


def evaluate_all(model, market, normalizer, bounds, config, output):
    summaries = {}

    attribution = []
    for policy in ("agent", "equal_weight", "cash"):
        result = evaluate(
            model,
            market,
            normalizer,
            bounds,
            config,
            policy,
            collect_attribution=(policy == "agent"),
        )
        if policy == "agent":
            rows, attribution = result
        else:
            rows = result
        write_rows(output / f"{policy}.csv", rows)
        summaries[policy] = performance(rows)

    components = [
        row for row in attribution
        if row["security_id"] in {"CASH", "TRADING_COSTS"}
    ]
    securities = [
        row for row in attribution
        if row["security_id"] not in {"CASH", "TRADING_COSTS"}
    ]
    securities.sort(key=lambda row: row["capital_contribution"])
    write_rows(output / "agent-attribution-all.csv", attribution)
    write_rows(output / "agent-attribution-worst.csv", securities[:25])
    write_rows(output / "agent-attribution-best.csv", securities[-25:][::-1])
    write_rows(output / "agent-attribution-components.csv", components)

    left, right = bounds
    benchmark_rows = [
        {
            "date": str(market.dates[index]),
            "return": float(market.benchmark[index]),
            "turnover": 0.0,
        }
        for index in range(left, right)
    ]
    write_rows(output / "benchmark.csv", benchmark_rows)
    summaries["SPY"] = performance(benchmark_rows)
    write_json(output / "metrics.json", summaries)

    return summaries


def save_checkpoint(path, model, normalizer, market, config, splits):
    temporary = path.with_suffix(".tmp")

    torch.save({
        "model": model.state_dict(),
        "config": asdict(config),
        "mean": normalizer.mean.tolist(),
        "scale": normalizer.scale.tolist(),
        "dataset_sha256": market.digest,
        "splits": splits,
        "research_assumptions": market.metadata.get("research_assumptions"),
        "model_architecture": (
            getattr(market, "model_architecture", "dirichlet-v1")
            + "+cross-sectional-ranks-v1"
        ),
    }, temporary)

    temporary.replace(path)


def train(args, market):
    config = Config(
        seed=args.seed,
        updates=args.updates,
        cost_bps=args.cost_bps,
        episode_steps=getattr(args, "episode_steps", 252),
        epochs=getattr(args, "epochs", 4),
        learning_rate=getattr(args, "learning_rate", 0.0003),
    )
    seed_everything(config.seed)

    splits = {
        "start": args.start,
        "validation_start": args.validation_start,
        "test_start": args.test_start,
        "end": args.end,
    }

    ordered = [
        np.datetime64(splits[key])
        for key in (
            "start", "validation_start", "test_start", "end"
        )
    ]

    if not all(a < b for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Split boundaries must increase")

    training = market.bounds(args.start, args.validation_start)
    validation = market.bounds(
        args.validation_start,
        args.test_start,
    )
    market.bounds(args.test_start, args.end)

    if training[1] - training[0] < config.episode_steps:
        raise ValueError("Training period needs at least 252 rows")

    normalizer = Normalizer.fit(market, training)
    model = getattr(market, "model_class", ActorCritic)(
        market.features.shape[-1]
        + extra_feature_count(market.features.shape[-1])
        + 3,
        config.hidden,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if (output / "best.pt").exists():
        raise ValueError("Choose a fresh output directory")

    write_json(output / "run.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": market.digest,
        "config": asdict(config),
        "splits": splits,
        "research_assumptions": market.metadata.get("research_assumptions"),
        "model_architecture": (
            getattr(market, "model_architecture", "dirichlet-v1")
            + "+cross-sectional-ranks-v1"
        ),
    })

    best_score = -float("inf")

    for update in range(1, config.updates + 1):
        left = random.randint(
            training[0],
            training[1] - config.episode_steps,
        )
        environment = getattr(market, "environment_class", Portfolio)(
            market,
            normalizer,
            left,
            left + config.episode_steps,
            config.cost_bps,
        )

        model.train()
        records = collect(model, environment, config)
        statistics = optimize(
            model, optimizer, records, config
        )

        report = {
            "update": update,
            "training_episode_return": environment.equity - 1,
            "loss": float(np.mean([
                row["loss"] for row in statistics
            ])),
        }

        if update == 1 or update % 10 == 0 or update == config.updates:
            model.eval()
            rows = evaluate(
                model,
                market,
                normalizer,
                validation,
                config,
                "agent",
            )
            metrics = performance(rows)
            score = metrics["total_return"] - metrics["max_drawdown"]
            report["validation"] = metrics

            if score > best_score:
                best_score = score
                save_checkpoint(
                    output / "best.pt",
                    model,
                    normalizer,
                    market,
                    config,
                    splits,
                )

            save_checkpoint(
                output / "latest.pt",
                model,
                normalizer,
                market,
                config,
                splits,
            )
            print(json.dumps(report), flush=True)

        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")

    checkpoint = torch.load(
        output / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    validation_output = output / "validation"
    validation_output.mkdir(exist_ok=True)

    evaluate_all(
        model,
        market,
        normalizer,
        validation,
        config,
        validation_output,
    )

    print(f"Best checkpoint: {output / 'best.pt'}")
    print("Training finished. Test period has not been evaluated.")


def test(args, market):
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["dataset_sha256"] != market.digest:
        raise ValueError("Checkpoint belongs to a different dataset")

    config = Config(**checkpoint["config"])
    normalizer = Normalizer(
        checkpoint["mean"],
        checkpoint["scale"],
    )
    model = getattr(market, "model_class", ActorCritic)(
        market.features.shape[-1]
        + extra_feature_count(market.features.shape[-1])
        + 3,
        config.hidden,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    splits = checkpoint["splits"]
    bounds = market.bounds(
        splits["test_start"],
        splits["end"],
    )

    output = args.checkpoint.parent / "test"

    if output.exists():
        raise ValueError("This run already has a test directory")

    output.mkdir()

    summaries = evaluate_all(
        model,
        market,
        normalizer,
        bounds,
        config,
        output,
    )
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)

    training = commands.add_parser("train")
    training.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/market.npz",
    )
    training.add_argument("--start", default="2001-01-01")
    training.add_argument(
        "--validation-start",
        default="2015-01-01",
    )
    training.add_argument("--test-start", default="2020-01-01")
    training.add_argument("--end", default="2026-08-19")
    training.add_argument("--updates", type=int, default=300)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--cost-bps", type=float, default=10)
    training.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/ppo-001",
    )

    testing = commands.add_parser("test")
    testing.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/market.npz",
    )
    testing.add_argument("--checkpoint", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "train":
        if args.updates < 1:
            parser.error("--updates must be positive")

        if not np.isfinite(args.cost_bps) or not (
            0 <= args.cost_bps <= 100
        ):
            parser.error("--cost-bps must be between 0 and 100")

    torch.set_num_threads(4)
    market = Market(args.data)

    if args.mode == "train":
        train(args, market)
    else:
        test(args, market)


if __name__ == "__main__":
    main()
