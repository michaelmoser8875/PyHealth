"""Load generated Normal/AS WAVs into PyHealth samples.

Paper:
    Nonaka et al. "Boosting Phonocardiogram Classification Performance with
    Function Generated Data." ML4H 2025 (PMLR 297).
    https://proceedings.mlr.press/v297/nonaka26a.html

Contributor:
    Tejaswi Sapkota (sapkota3@illinois.edu)

Description:
    Reads the generator's manifest without changing its partitions. Uses the
    signal/label schemas from Michael's BMDHSBinaryClassification task, which
    are compatible with Franko's ResNet1D. Only training fits processors.
    Reading CSV samples needs no PyHealth installation; creating datasets needs
    the local PyHealth 2.x checkout and its matching audio dependencies.
"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

# Keep these identical to the real-data task. Model-side normalization is separate.
INPUT_SCHEMA = {"signal": ("audio", {"sample_rate": 8000, "duration": 10.0})}
OUTPUT_SCHEMA = {"label": "binary"}
CLASS_LABELS = {"Normal": 0, "AS": 1}


def read_synthetic_manifest(
    manifest_path: str | Path,
    limit_per_class: int | None = None,
    verify_checksums: bool = False,
) -> dict[str, list[dict]]:
    """Read training/validation samples and resolve WAV paths.

    Checks all rows before returning a limited subset. Generated patient IDs,
    recording IDs, paths, and seeds must be unique across both partitions.

    Args:
        manifest_path: manifest.csv produced by generate_dataset.
        limit_per_class: Optional positive limit per class in each partition.
            Use a small limit for integration checks; None loads all CSV rows.
        verify_checksums: Read each WAV and compare its SHA-256 with the CSV.

    Returns:
        Dictionary with train and validation sample lists. Each sample contains
        absolute signal path, binary label, patient_id, record_id and seed.

    Examples:
        >>> samples = read_synthetic_manifest(  # doctest: +SKIP
        ...     "synthetic_data/paper_seed42/manifest.csv", limit_per_class=2
        ... )
    """
    if limit_per_class is not None and (
        type(limit_per_class) is not int or limit_per_class < 1
    ):
        raise ValueError("limit_per_class must be a positive integer or None.")
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    settings_path = root / "dataset.json"
    settings = None
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        if settings.get("status") != "complete":
            raise ValueError("Dataset generation is not complete.")
    partitions = {"train": [], "validation": []}
    counts = {(split, label): 0 for split in partitions for label in CLASS_LABELS}
    seen = {key: set() for key in ("recording_id", "patient_id", "signal", "seed")}
    required = {
        "recording_id",
        "patient_id",
        "split",
        "class_name",
        "label",
        "signal",
        "seed",
        "sample_rate",
        "samples",
        "duration_seconds",
        "sha256",
    }
    with manifest_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest is missing required generator columns.")
        for line_number, row in enumerate(reader, start=2):
            try:
                split, class_name = row["split"], row["class_name"]
                if split not in partitions or class_name not in CLASS_LABELS:
                    raise ValueError("Unsupported partition or class.")
                label = int(row["label"])
                seed = int(row["seed"])
                if label != CLASS_LABELS[class_name] or seed < 0:
                    raise ValueError("Incorrect binary label or negative seed.")
                if (
                    int(row["sample_rate"]),
                    int(row["samples"]),
                    float(row["duration_seconds"]),
                ) != (8000, 80000, 10.0):
                    raise ValueError("Expected mono 10-second audio at 8 kHz.")
                relative_path = Path(row["signal"])
                path = (root / relative_path).resolve()
                if relative_path.is_absolute() or not path.is_relative_to(root):
                    raise ValueError("Audio path must stay inside the dataset folder.")
                if not path.is_file():
                    raise ValueError(f"Missing audio: {row['signal']}")
                values = {
                    "recording_id": row["recording_id"].strip(),
                    "patient_id": row["patient_id"].strip(),
                    "signal": str(path),
                    "seed": seed,
                }
                for key, value in values.items():
                    if value == "" or value in seen[key]:
                        raise ValueError(f"Empty or duplicate {key}.")
                    seen[key].add(value)
                if verify_checksums:
                    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                    if checksum != row["sha256"]:
                        raise ValueError(f"Checksum mismatch: {row['signal']}")
                key = (split, class_name)
                counts[key] += 1
                if limit_per_class is None or counts[key] <= limit_per_class:
                    partitions[split].append(
                        {
                            "patient_id": values["patient_id"],
                            "record_id": values["recording_id"],
                            "signal": str(path),
                            "label": label,
                            "seed": seed,
                        }
                    )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Manifest row {line_number}: {error}") from error
    if any(count == 0 for count in counts.values()):
        raise ValueError("Training and validation must each contain Normal and AS.")
    if settings is not None:
        if settings.get("completed_total") != sum(counts.values()):
            raise ValueError("Manifest count differs from completed dataset total.")
        for (split, class_name), count in counts.items():
            if count != settings.get("counts_per_class", {}).get(split):
                raise ValueError(f"Incorrect manifest count for {split}/{class_name}.")
    return partitions


def load_synthetic_datasets(
    manifest_path: str | Path,
    limit_per_class: int | None = None,
    in_memory: bool = False,
    verify_checksums: bool = False,
) -> tuple:
    """Create PyHealth training/validation datasets using the real-task schemas.

    Training fits the audio and label processors. Validation reuses those exact
    processors, without fitting again. Defaults to disk-backed datasets so the
    full 12,000 waveforms are not all retained in Python memory. Construction
    still decodes audio and writes processed data to PyHealth's temporary cache.

    Args:
        manifest_path: Path to the generated dataset's manifest.csv.
        limit_per_class: Optional small subset for checking the loader.
        in_memory: Use True only for small examples that fit in memory.
        verify_checksums: Verify stored WAV checksums while reading the CSV.

    Returns:
        train_dataset, validation_dataset. Real test data is loaded separately.

    Examples:
        >>> train, validation = load_synthetic_datasets(  # doctest: +SKIP
        ...     "synthetic_data/paper_seed42/manifest.csv",
        ...     limit_per_class=2, in_memory=True,
        ... )
    """
    samples = read_synthetic_manifest(
        manifest_path,
        limit_per_class=limit_per_class,
        verify_checksums=verify_checksums,
    )
    return _create_synthetic_datasets(samples, in_memory)


def _create_synthetic_datasets(samples, in_memory=False):
    """Shared PyHealth construction for source manifests and run split plans."""
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError(
            "PyHealth 2.x requires Python 3.12 or 3.13. "
            "Use a separate compatible environment; CSV reading works without it."
        )
    try:
        from pyhealth.datasets import create_sample_dataset
    except ImportError as error:
        raise ImportError(
            "Install the team's PyHealth checkout and its audio dependencies "
            "in this Python environment before creating SampleDatasets."
        ) from error
    train = create_sample_dataset(
        samples["train"],
        copy.deepcopy(INPUT_SCHEMA),
        copy.deepcopy(OUTPUT_SCHEMA),
        dataset_name="synthetic_pcg_train",
        task_name="BMDHSBinaryClassification",
        in_memory=in_memory,
    )
    validation = create_sample_dataset(
        samples["validation"],
        copy.deepcopy(INPUT_SCHEMA),
        copy.deepcopy(OUTPUT_SCHEMA),
        dataset_name="synthetic_pcg_validation",
        task_name="BMDHSBinaryClassification",
        input_processors=train.input_processors,
        output_processors=train.output_processors,
        in_memory=in_memory,
    )
    return train, validation
