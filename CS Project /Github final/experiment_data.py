"""Load saved assignments through existing team interfaces, without editing them.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: https://proceedings.mlr.press/v297/nonaka26a.html
"""

from collections import Counter
from pathlib import Path

import numpy as np

from synthetic_splits import load_synthetic_run, read_synthetic_run
from workflow import (digest, read_json, require, resolve_from,
                      validate_real_plan)


def setup_info(workspace):
    workspace = Path(workspace).resolve()
    setup = read_json(workspace / "data_setup.json")
    require(read_json(workspace / "setup_status.json")["status"] == "passed",
            "Dataset setup did not finish successfully")
    real_path = workspace / setup["real_plan"]
    syn_path = workspace / setup["synthetic_plan"]
    require(digest(real_path) == setup["real_plan_sha256"], "Real plan changed")
    require(digest(syn_path) == setup["synthetic_plan_sha256"], "Synthetic plan changed")
    real, synthetic = read_json(real_path), read_json(syn_path)
    validate_real_plan(real)
    root = resolve_from(real["dataset_root"], real_path)
    require(digest(root / "train.csv") == real["source_csv_sha256"], "Real CSV changed")
    require(digest(root / "additional_metadata.csv") == setup["metadata_csv_sha256"],
            "Real metadata CSV changed")
    checksum_path = workspace / "plans/real_audio_checksums.json"
    require(digest(checksum_path) == setup["real_audio_checksums_sha256"],
            "Real audio checksum index changed")
    for relative, expected in read_json(checksum_path).items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and digest(path) == expected,
                f"Real audio changed: {relative}")
    require(len(synthetic["runs"]) == 5 and all(
        real["runs"][i]["training_seed"] == synthetic["runs"][i]["training_seed"]
        for i in range(5)), "Real/synthetic run pairing differs")
    return setup, real, root, syn_path


def subset_patients(dataset, patients, count, seed):
    """Smoke-only stratified patient subset; never moves patients across splits."""
    if count is None:
        return dataset
    rng = np.random.default_rng(seed)
    keep = []
    for label in (0, 1):
        ids = sorted(str(pid) for pid in dataset.patient_to_index
                     if patients[str(pid)]["label"] == label)
        require(len(ids) >= count, "Not enough patients for smoke subset")
        keep.extend(rng.permutation(ids).tolist()[:count])
    indices = sorted(i for pid in keep for i in dataset.patient_to_index[pid])
    return dataset.subset(indices)


def load_real(workspace, index, patient_limit=None):
    from pyhealth.datasets import BMDHSDataset, SubjectSplit
    from pyhealth.tasks import BMDHSBinaryClassification

    _, plan, root, _ = setup_info(workspace)
    run = plan["runs"][index]
    base = BMDHSDataset(root=str(root),
                        cache_dir=str(Path(workspace) / "cache/real"), num_workers=1)
    parts = base.set_task(BMDHSBinaryClassification(**plan["task"]), split=SubjectSplit(
        subjects=tuple(run[k] for k in ("train", "validation", "test"))))
    for name, part in zip(("train", "validation", "test"), parts):
        require(set(map(str, part.patient_to_index)) == set(run[name]),
                f"Real {name} patient assignments differ")
        require(len(part) == sum(len(plan["patients"][pid]["recordings"])
                                for pid in run[name]), f"Real {name} count differs")
    train = subset_patients(parts[0], plan["patients"], patient_limit, index)
    validation = subset_patients(parts[1], plan["patients"], patient_limit, index + 100)
    counts = Counter()
    for pid in train.patient_to_index:
        counts[plan["patients"][str(pid)]["label"]] += len(train.patient_to_index[pid])
    require(counts[0] > 0 and counts[1] > 0, "Real training needs both classes")
    return train, validation, parts[2], counts[0] / counts[1]


def load_run(workspace, index, patient_limit=None):
    _, real, _, syn_path = setup_info(workspace)
    rows = read_synthetic_run(syn_path, run_index=index, verify_checksums=True)
    require(all(row["patient_id"] not in real["patients"]
                for values in rows.values() for row in values), "Domain ID collision")
    train, validation, test, weight = load_real(workspace, index, patient_limit)
    syn_train, syn_val = load_synthetic_run(syn_path, run_index=index, in_memory=False)
    for ds in (validation, test, syn_train, syn_val):
        require(ds.input_schema == train.input_schema and
                ds.output_schema == train.output_schema, "Domain schema mismatch")
    for name, ds in (("train", syn_train), ("validation", syn_val)):
        require(len(ds) == len(rows[name]), "Synthetic processed count differs")
        require(set(map(str, ds.patient_to_index)) == {
            str(row["patient_id"]) for row in rows[name]}, "Synthetic IDs differ")
        for label in (0, 1):
            i = next(i for i, row in enumerate(rows[name]) if row["label"] == label)
            item = ds[i]
            require(int(item["label"].item()) == label, "Synthetic label mapping differs")
    return {"real_train": train, "real_validation": validation,
            "syn_train": syn_train, "syn_validation": syn_val,
            "real_weight": weight}


class IndexView:
    """Map-style order for deterministic loaders and repeated Mixed cycles."""

    def __init__(self, base, indices):
        self.base, self.indices = base, list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


def shuffled_cycles(size, total, seed):
    """Sample without replacement within each seeded cycle; cycle if necessary."""
    require(size > 0 and total >= 0, "Invalid sampling dimensions")
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < total:
        result.extend(rng.permutation(size).tolist()[:total - len(result)])
    return result


def loader(dataset, batch_size, seed=0, shuffle=False, indices=None):
    from pyhealth.datasets import get_dataloader

    if indices is None:
        indices = (np.random.default_rng(seed).permutation(len(dataset)).tolist()
                   if shuffle else range(len(dataset)))
    return get_dataloader(IndexView(dataset, indices), batch_size=batch_size,
                          shuffle=False, seed=seed, num_workers=0, drop_last=False)
