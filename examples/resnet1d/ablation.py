"""ResNet1D ablations using Group 1's patient-separated PyHealth datasets."""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ResNet1D
from pyhealth.trainer import Trainer


def _same_state(left, right):
    """Compare processor state across separately loaded copies."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right)
        )
    if hasattr(left, "__dict__"):
        return _same_state(vars(left), vars(right))
    return left == right


def check_splits(train, validation, test):
    """Check supplied splits for missing IDs, empty splits, and patient overlap."""
    groups = []
    for split in (train, validation, test):
        # Use PyHealth's index to avoid decoding held-out audio just to read IDs.
        mapping = getattr(split, "patient_to_index", None)
        values = (
            mapping.keys()
            if mapping is not None
            else (row.get("patient_id") for row in split)
        )
        ids = set()
        for value in values:
            if value is None or str(value).strip().lower() in {"", "none", "nan"}:
                raise ValueError("Every split requires nonempty patient IDs")
            ids.add(str(value))
        if not ids:
            raise ValueError("Every split requires nonempty patient IDs")
        groups.append(ids)
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("Patient leakage across supplied splits")


def run_ablation(
    train,
    validation,
    test,
    output,
    smoke=False,
    seeds=(0, 1, 2),
    epochs=20,
    batch_size=8,
    device="cpu",
):
    """Train fresh models and save validation results on fixed patient splits.

    Arguments are processed PyHealth datasets, e.g. supplied by Group 1.
    Validation and test must reuse the training-fitted processors. The test
    split is inspected only for patient separation. F1 uses threshold 0.5.
    """
    check_splits(train, validation, test)
    for name in (
        "input_schema",
        "output_schema",
        "input_processors",
        "output_processors",
    ):
        if not _same_state(getattr(train, name), getattr(validation, name)):
            raise ValueError(
                f"Validation {name} differs from training; "
                "reuse training-fitted processors"
            )
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a nonempty sequence of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier runs")
    output.mkdir(parents=True, exist_ok=True)
    grid = (
        [(1e-4, 4, (1, 1))]
        if smoke
        else itertools.product((2e-5, 1e-4), (32, 64), ((2, 2, 2, 2), (3, 4, 6, 3)))
    )
    results = []
    for lr, width, layers in grid:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
            config = {"base_width": width, "layers": layers}
            model = ResNet1D(train, **config)
            loader = get_dataloader(train, batch_size, shuffle=True, seed=seed)
            val_loader = get_dataloader(validation, batch_size, shuffle=False)
            trainer = Trainer(
                model=model,
                device=device,
                enable_logging=False,
                metrics=["roc_auc", "pr_auc", "f1"],
            )
            start = time.perf_counter()
            trainer.train(
                loader,
                val_loader,
                epochs=epochs,
                optimizer_params={"lr": lr},
                monitor="roc_auc",
            )
            scores = trainer.evaluate(val_loader)
            name = f"run_{len(results):03d}"
            torch.save(model.state_dict(), output / f"{name}.pt")
            results.append(
                {
                    "run": name,
                    "smoke": smoke,
                    "seed": seed,
                    "lr": lr,
                    "model": config,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "device": device,
                    "torch_version": torch.__version__,
                    "f1_threshold": 0.5,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "seconds": time.perf_counter() - start,
                    "validation": scores,
                }
            )
            (output / "results.json").write_text(json.dumps(results, indent=2))
    return results


def smoke_datasets():
    """Build small random tensor datasets for software checks."""
    torch.manual_seed(0)
    datasets = []
    for part in range(3):
        samples = [
            {
                "patient_id": f"{part}_{i}",
                "signal": torch.randn(1, 128).tolist(),
                "label": i % 2,
            }
            for i in range(4)
        ]
        options = (
            {}
            if not datasets
            else {
                "input_processors": datasets[0].input_processors,
                "output_processors": datasets[0].output_processors,
            }
        )
        datasets.append(
            create_sample_dataset(
                samples, {"signal": "tensor"}, {"label": "binary"}, **options
            )
        )
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_ablation(*smoke_datasets(), args.output, smoke=True, seeds=(0,), epochs=1)
