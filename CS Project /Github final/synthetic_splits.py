"""Reuse one synthetic audio pool across configurable training/validation runs.

Only recording IDs and seeds are written. Audio and the source manifest remain
unchanged. Run ``python synthetic_splits.py --help`` for the command-line API.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from synthetic_manifest import read_synthetic_manifest


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _seeds(values, count, start, name):
    values = list(range(start, start + count)) if values is None else list(values)
    if len(values) != count:
        raise ValueError(f"{name} requires exactly {count} values.")
    for value in values:
        _positive_int(value, name, minimum=0)
    if len(set(values)) != count:
        raise ValueError(f"{name} must be distinct across runs.")
    return values


def create_split_plan(
    manifest_path,
    output_path,
    n_runs=5,
    split_seed=42,
    split_seeds=None,
    training_seeds=None,
    preserve_first=True,
    train_per_class=None,
    validation_per_class=None,
):
    """Save deterministic stratified splits of the full existing audio pool.

    Defaults preserve the original manifest for run 1 and shuffle the pool for
    runs 2-5 using split seeds 42-45. Training seeds default to 0-4; they are
    metadata for the training runner, not applied to a model by this function.

    Counts default to the source manifest's per-class counts. Counts must use
    the whole balanced pool. With 6,000 recordings per class, the default source
    counts of 5,000/1,000 reproduce the paper's per-run sizes.

    split_seeds contains one seed per reshuffled run (n_runs-1 when preserving
    the first run, otherwise n_runs). Explicit seeds override split_seed.
    Set preserve_first=False to reshuffle every run or change partition sizes.
    The output JSON must not exist; source audio/metadata are never rewritten.
    """
    _positive_int(n_runs, "n_runs")
    _positive_int(split_seed, "split_seed", minimum=0)
    if type(preserve_first) is not bool:
        raise ValueError("preserve_first must be a boolean.")
    split_seeds = _seeds(
        split_seeds, n_runs - int(preserve_first), split_seed, "split_seeds"
    )
    training_seeds = _seeds(training_seeds, n_runs, 0, "training_seeds")
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("Split plan already exists; choose a new output path.")
    source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    partitions = read_synthetic_manifest(manifest_path)
    by_label = {
        label: sorted(
            [s["record_id"] for samples in partitions.values() for s in samples
             if s["label"] == label]
        )
        for label in (0, 1)
    }
    if len(by_label[0]) != len(by_label[1]):
        raise ValueError("The source audio pool must be balanced across classes.")
    original_counts = {
        split: [sum(s["label"] == label for s in samples) for label in (0, 1)]
        for split, samples in partitions.items()
    }
    if any(counts[0] != counts[1] for counts in original_counts.values()):
        raise ValueError("The original partitions must each be class balanced.")
    size = len(by_label[0])
    if train_per_class is None and validation_per_class is None:
        train_per_class = original_counts["train"][0]
        validation_per_class = original_counts["validation"][0]
    elif train_per_class is None:
        _positive_int(validation_per_class, "validation_per_class")
        train_per_class = size - validation_per_class
    elif validation_per_class is None:
        _positive_int(train_per_class, "train_per_class")
        validation_per_class = size - train_per_class
    _positive_int(train_per_class, "train_per_class")
    _positive_int(validation_per_class, "validation_per_class")
    if train_per_class + validation_per_class != size:
        raise ValueError("Training + validation counts must use the entire pool.")
    if preserve_first and train_per_class != original_counts["train"][0]:
        raise ValueError("Changed counts require preserve_first=False.")

    runs = []
    for index in range(n_runs):
        preserved = preserve_first and index == 0
        seed = None if preserved else split_seeds[index - int(preserve_first)]
        if preserved:
            ids = {split: sorted(s["record_id"] for s in samples)
                   for split, samples in partitions.items()}
        else:
            rng = np.random.default_rng(seed)
            ids = {"train": [], "validation": []}
            for label in (0, 1):
                shuffled = rng.permutation(by_label[label]).tolist()
                ids["train"].extend(shuffled[:train_per_class])
                ids["validation"].extend(shuffled[train_per_class:])
            ids = {split: sorted(values) for split, values in ids.items()}
        runs.append({
            "run_index": index,
            "run_name": f"run_{index + 1:02d}",
            "split_seed": seed,
            "training_seed": training_seeds[index],
            "preserves_original": preserved,
            **ids,
        })
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source manifest changed while building the split plan.")
    plan = {
        "version": 1,
        "policy": "fixed_pool_stratified_resplits",
        "source_manifest": os.path.relpath(manifest_path, output_path.parent),
        "source_manifest_sha256": source_hash,
        "counts_per_class": {"train": train_per_class,
                             "validation": validation_per_class},
        "pool_per_class": size,
        "numpy_version": np.__version__,
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    return output_path


def read_synthetic_run(plan_path, run_index=0, limit_per_class=None,
                       verify_checksums=False):
    """Validate all run assignments, then return one run's sample partitions.

    The plan's source manifest path is relative to the JSON. Preserve their
    relative layout when moving files. Full source manifest validation still
    runs, including path containment and optional audio checksum verification.
    Returned sample dictionaries are compatible with the existing PyHealth loader.
    """
    _positive_int(run_index, "run_index", minimum=0)
    if limit_per_class is not None:
        _positive_int(limit_per_class, "limit_per_class")
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("version") != 1 or plan.get("policy") != "fixed_pool_stratified_resplits":
        raise ValueError("Unsupported synthetic split plan.")
    source = (plan_path.parent / plan["source_manifest"]).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan["source_manifest_sha256"]:
        raise ValueError("Source manifest checksum differs from the split plan.")
    original = read_synthetic_manifest(source, verify_checksums=verify_checksums)
    pool = {s["record_id"]: s for samples in original.values() for s in samples}
    runs = plan.get("runs", [])
    if not runs or run_index >= len(runs):
        raise ValueError("run_index is outside the split plan.")
    for split in ("train", "validation"):
        _positive_int(plan["counts_per_class"][split], f"{split} count")
    for index, run in enumerate(runs):
        if run.get("run_index") != index:
            raise ValueError("Run indices must be consecutive from zero.")
        _positive_int(run["training_seed"], "training_seed", minimum=0)
        if run["split_seed"] is not None:
            _positive_int(run["split_seed"], "split_seed", minimum=0)
        train, val = run["train"], run["validation"]
        if (len(train) != len(set(train)) or len(val) != len(set(val))
                or set(train) & set(val) or set(train) | set(val) != set(pool)):
            raise ValueError("Run IDs must cover the pool exactly without overlap.")
        for split, ids in (("train", train), ("validation", val)):
            for label in (0, 1):
                count = sum(pool[rid]["label"] == label for rid in ids)
                if count != plan["counts_per_class"][split]:
                    raise ValueError("Run class counts differ from the split plan.")
        if run.get("preserves_original"):
            if any(set(run[split]) != {s["record_id"] for s in original[split]}
                   for split in ("train", "validation")):
                raise ValueError("Preserved run differs from the original partition.")
    selected = runs[run_index]
    result = {}
    for split in ("train", "validation"):
        counts = {0: 0, 1: 0}
        result[split] = []
        for rid in selected[split]:
            sample = pool[rid]
            label = sample["label"]
            if limit_per_class is None or counts[label] < limit_per_class:
                result[split].append(dict(sample))
                counts[label] += 1
    return result


def load_synthetic_run(plan_path, run_index=0, limit_per_class=None,
                       in_memory=False, verify_checksums=False):
    """Create PyHealth datasets for a run; fit processors on its training only.

    The caller must separately apply the recorded training_seed to its training
    runner and reuse this run across Synthetic, Syn-to-Real and Mixed conditions.
    """
    from synthetic_manifest import _create_synthetic_datasets

    samples = read_synthetic_run(plan_path, run_index, limit_per_class,
                                 verify_checksums)
    return _create_synthetic_datasets(samples, in_memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-seeds", type=int, nargs="+")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--reshuffle-first", action="store_true")
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    args = parser.parse_args()
    path = create_split_plan(
        args.manifest, args.output, n_runs=args.runs, split_seed=args.split_seed,
        split_seeds=args.split_seeds, training_seeds=args.training_seeds,
        preserve_first=not args.reshuffle_first,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
    )
    print(f"Saved {args.runs} runs: {path}")


if __name__ == "__main__":
    main()
