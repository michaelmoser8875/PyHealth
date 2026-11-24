"""Portable configuration and provenance helpers; no training on import.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: Nonaka et al., Boosting Phonocardiogram Classification Performance
with Function Generated Data. https://proceedings.mlr.press/v297/nonaka26a.html
"""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOLS = ("real", "syn", "syn2real", "mixed")
PRESETS = {
    "smoke": dict(train_per_class=10, validation_per_class=10, runs=1,
                  batch_size=4, epochs=2, val_every=1, patience=5,
                  real_patients_per_class=2, minutes=0),
    "pilot": dict(train_per_class=500, validation_per_class=100, runs=1,
                  batch_size=8, epochs=100, val_every=1, patience=5,
                  real_patients_per_class=None, minutes=60),
    "full": dict(train_per_class=5000, validation_per_class=1000, runs=5,
                 batch_size=512, epochs=500, val_every=5, patience=5,
                 real_patients_per_class=None, minutes=0),
}
STAGE_FRACTIONS = {"real": 0.2, "syn": 0.4, "syn2real": 0.2, "mixed": 0.2}
DEVIATIONS = [
    "Team preprocessing: mono 8kHz, first10s crop/zero padding, model-side "
    "z-score. Pinned SynPCG classifier uses 1kHz/5s, normalization before "
    "random training crop/center evaluation crop.",
    "No training augmentation. The paper uses six augmentations; this package "
    "deliberately leaves teammates' task/model code unchanged.",
    "AS-versus-Normal excludes other-disease-only patients. Pinned reference "
    "code uses AS-flag negatives including other diseases; cohort/splits differ.",
    "One synthetic pool with five resplits, not five independently generated pools.",
    "Generator RNG, rejection, time-grid and float32 export differences are "
    "described in GENERATOR_README.md.",
    "Franko's model reuses PyHealth's backbone; reference initialization and "
    "checkpoint naming differ. Reference checkpoints are not directly loadable.",
    "Mixed is a new extension. A Mixed epoch exposes each real example once "
    "plus an equal number of synthetic examples; epochs are not compute-matched.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    names = ("workflow.py", "prepare_data.py", "experiment_data.py",
             "run_experiments.py", "evaluate_results.py", "synthetic_pcg.py",
             "synthetic_manifest.py", "synthetic_splits.py",
             "real_split_template.json")
    return {name: digest(HERE / name) for name in names}


def resolve_from(path, location):
    return (Path(location).parent / path).resolve()


def validate_real_plan(plan):
    require(plan["task"] == {"disease": "AS", "other_disease_policy": "exclude"},
            "This package implements the frozen AS/exclude task only")
    pool = set(plan["patients"])
    require(len(plan["runs"]) == 5, "Expected five saved real runs")
    fixed_test = None
    for index, run in enumerate(plan["runs"]):
        require(run["run_index"] == index and run["training_seed"] == index,
                "Unexpected run index/training seed")
        groups = [set(run[k]) for k in ("train", "validation", "test")]
        require(all(len(g) == len(run[k]) for g, k in zip(
            groups, ("train", "validation", "test"))), "Duplicate patients")
        require(not any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))),
                "Real patient leakage")
        require(set.union(*groups) == pool, "Real split does not cover cohort")
        require(all({plan["patients"][pid]["label"] for pid in g} == {0, 1}
                    for g in groups), "Every real partition needs both classes")
        if fixed_test is None:
            fixed_test = groups[2]
        require(groups[2] == fixed_test, "Real test membership changed across runs")


def settings(mode, batch_size=None, epochs=None, val_every=None,
             patience=None, minutes=None, run_indices=None):
    require(mode in PRESETS, "Unknown mode")
    value = dict(PRESETS[mode])
    for name, override in (("batch_size", batch_size), ("epochs", epochs),
                           ("val_every", val_every), ("patience", patience),
                           ("minutes", minutes)):
        if override is not None:
            value[name] = override
    for name in ("batch_size", "epochs", "val_every", "patience"):
        require(type(value[name]) is int and value[name] > 0, f"Invalid {name}")
    require(value["batch_size"] >= 2 and value["batch_size"] % 2 == 0,
            "Batch size must be even and >=2 for balanced Mixed batches")
    require(math.isfinite(value["minutes"]) and value["minutes"] >= 0,
            "Minutes must be finite/nonnegative")
    indices = list(range(value["runs"])) if run_indices is None else run_indices
    require(bool(indices) and len(indices) == len(set(indices)) and
            all(type(i) is int and 0 <= i < 5 for i in indices), "Invalid run indices")
    value.update(mode=mode, run_indices=indices, lr=2e-5, augmentation="none",
                 model={"layers": [2, 2, 2, 2], "base_width": 64,
                        "normalize": True})
    return value
