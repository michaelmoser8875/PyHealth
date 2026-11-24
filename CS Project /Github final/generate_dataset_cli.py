"""Command-line entry point for the Normal/AS synthetic PCG generator."""

import argparse
from pathlib import Path

from synthetic_pcg import generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty dataset directory.")
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--validation-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing", choices=("reference", "positioned_s1"),
                        default="reference")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        master_seed=args.seed,
        timing=args.timing,
    )
    print(f"Dataset index: {manifest}")


if __name__ == "__main__":
    main()
