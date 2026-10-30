import argparse
from pathlib import Path

from .research_data import ROOT, load_market, prepare


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    builder = commands.add_parser("prepare")
    builder.add_argument("--staging", type=Path, default=ROOT / "outputs/training-staging-v3")
    builder.add_argument("--source", type=Path, default=ROOT / "data/raw/final-extended")
    builder.add_argument("--output", type=Path, default=ROOT / "data/processed/research-stress")
    builder.add_argument("--unknown-recovery", type=float, default=0.0)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=ROOT / "data/processed/research-stress")
    training.add_argument("--output", type=Path, default=ROOT / "outputs/research-ppo-001")
    training.add_argument("--start", default="1996-01-02")
    training.add_argument("--validation-start", default="2015-01-01")
    training.add_argument("--test-start", default="2020-01-01")
    training.add_argument("--end", default="2026-10-27")
    training.add_argument("--updates", type=int, default=50)
    training.add_argument("--episode-steps", type=int, default=252)
    training.add_argument("--epochs", type=int, default=2)
    training.add_argument("--learning-rate", type=float, default=.0003)
    training.add_argument("--cost-bps", type=float, default=10)
    training.add_argument("--seed", type=int, default=7)
    testing = commands.add_parser("test")
    testing.add_argument("--data", type=Path, default=ROOT / "data/processed/research-stress")
    testing.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.staging, args.source, args.output, args.unknown_recovery)
        return
    import numpy as np
    import torch
    from .train import test, train

    torch.set_num_threads(4)
    if args.mode == "train":
        if args.updates < 1 or args.episode_steps < 2 or args.epochs < 1:
            parser.error("Training counts must be positive and episodes need at least two steps")
        if not np.isfinite(args.cost_bps) or not 0 <= args.cost_bps <= 100:
            parser.error("Cost must be between zero and 100 basis points")
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("Learning rate must be finite and positive")
    market = load_market(args.data)
    if args.mode == "train":
        train(args, market)
    else:
        test(args, market)


if __name__ == "__main__":
    main()
