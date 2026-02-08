"""CLI entry point for collecting random rollout data."""

import argparse
import sys
from pathlib import Path

# Add experiment root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data_collector import collect_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CarRacing random rollouts")
    parser.add_argument(
        "--num-episodes", type=int, default=None, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per episode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = Config()
    if args.num_episodes is not None:
        config.data.num_episodes = args.num_episodes
    if args.max_steps is not None:
        config.data.max_steps = args.max_steps
    if args.output_dir is not None:
        config.data.data_dir = Path(args.output_dir)

    collect_data(config.data, env_name=config.env_name, base_seed=args.seed)


if __name__ == "__main__":
    main()
