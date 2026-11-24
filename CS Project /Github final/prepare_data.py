"""Download/reuse real data, generate synthetic WAVs, and save portable plans.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: https://proceedings.mlr.press/v297/nonaka26a.html
Run manually; dataset creation is never triggered by importing this module.
"""

import argparse
import csv
import os
from pathlib import Path
import subprocess

from synthetic_manifest import read_synthetic_manifest
from synthetic_pcg import generate_dataset
from synthetic_splits import create_split_plan, read_synthetic_run
from workflow import (HERE, PRESETS, digest, read_json, require,
                      validate_real_plan, write_json)

DATASET_URL = "https://github.com/sani002/BMD-HS-Dataset.git"
DATASET_REVISION = "4745751855d15ff5fc9aaed29c32ad3ec7a5d409"


def audit_real(root, plan):
    """Validate frozen eligible cohort against source CSV and existing paths."""
    require((root / "additional_metadata.csv").is_file(), "Missing metadata CSV")
    require(digest(root / "train.csv") == plan["source_csv_sha256"],
            "BMD-HS CSV differs from frozen cohort. Use the pinned dataset revision; "
            "do not silently regenerate the real split")
    observed, excluded = {}, []
    with (root / "train.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["patient_id"] for r in rows]
    require(len(ids) == len(set(ids)), "Duplicate real patient IDs")
    for row in rows:
        pid = row["patient_id"]
        if row["N"].strip().upper() == "N/A":
            excluded.append(pid)
            continue
        positive = int(float(row["AS"] or 0)) == 1
        normal = int(float(row["N"] or 0)) == 1
        if not positive and not normal:
            excluded.append(pid)
            continue
        recordings = []
        for slot in range(1, 9):
            name = row.get(f"recording_{slot}", "").strip()
            if not name:
                continue
            path = root / "train" / f"{name}.wav"
            require(path.resolve().is_relative_to(root.resolve()), "Unsafe audio path")
            require(path.is_file(), f"Missing eligible recording: {path}")
            recordings.append(path.relative_to(root).as_posix())
        observed[pid] = {"label": int(positive), "recordings": recordings}
    require(observed == plan["patients"], "Real cohort differs from saved assignments")
    checksums = {relative: digest(root / relative)
                 for patient in observed.values() for relative in patient["recordings"]}
    return {"all_patients": len(rows), "eligible_patients": len(observed),
            "eligible_recordings": sum(len(p["recordings"]) for p in observed.values()),
            "excluded_patient_count": len(excluded), "audio_checksums": checksums}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=PRESETS, required=True)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="New mode-specific directory outside Git")
    parser.add_argument("--existing-real", type=Path,
                        help="Reuse a BMD-HS checkout instead of downloading")
    parser.add_argument("--existing-synthetic-manifest", type=Path,
                        help="Reuse a validated pool with this mode's exact counts")
    parser.add_argument("--verify-audio-checksums", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    require(not workspace.exists(), "Workspace already exists. Choose a new directory; "
            "existing data are not overwritten. Existing prepared workspaces can be "
            "used directly by run_experiments.py")
    workspace.mkdir(parents=True)
    write_json(workspace / "setup_status.json", {"status": "running", "mode": args.mode})
    try:
        real = args.existing_real.resolve() if args.existing_real else workspace / "real"
        if args.existing_real is None:
            print("Downloading pinned BMD-HS source...", flush=True)
            subprocess.run(["git", "clone", DATASET_URL, str(real)], check=True)
            subprocess.run(["git", "-C", str(real), "checkout", "--detach",
                            DATASET_REVISION], check=True)
        template = read_json(HERE / "real_split_template.json")
        validate_real_plan(template)
        audit = audit_real(real, template)
        plans = workspace / "plans"
        plans.mkdir()
        write_json(plans / "real_audio_checksums.json", audit.pop("audio_checksums"))
        template["dataset_root"] = os.path.relpath(real, plans)
        write_json(plans / "real.json", template)
        preset = PRESETS[args.mode]
        manifest = (args.existing_synthetic_manifest.resolve()
                    if args.existing_synthetic_manifest else None)
        if manifest is None:
            print("Generating fresh synthetic pool...", flush=True)
            manifest = generate_dataset(
                workspace / "synthetic", train_per_class=preset["train_per_class"],
                validation_per_class=preset["validation_per_class"], master_seed=42)
        rows = read_synthetic_manifest(manifest,
                                        verify_checksums=args.verify_audio_checksums)
        generation = read_json(manifest.parent / "dataset.json")
        require(generation["generator_sha256"] == digest(HERE / "synthetic_pcg.py") and
                generation["timing"] == "reference" and generation["master_seed"] == 42,
                "Reused synthetic data must use this generator, seed42 and reference timing")
        for split, count in (("train", preset["train_per_class"]),
                             ("validation", preset["validation_per_class"])):
            require(all(sum(r["label"] == label for r in rows[split]) == count
                        for label in (0, 1)), "Synthetic counts do not match mode")
        create_split_plan(manifest, plans / "synthetic.json", n_runs=5,
                          training_seeds=[0, 1, 2, 3, 4], split_seeds=[42, 43, 44, 45])
        for index in range(5):
            read_synthetic_run(plans / "synthetic.json", run_index=index)
        config = {"version": 1, "mode": args.mode,
                  "real_plan": "plans/real.json", "synthetic_plan": "plans/synthetic.json",
                  "real_plan_sha256": digest(plans / "real.json"),
                  "synthetic_plan_sha256": digest(plans / "synthetic.json"),
                  "real_csv_sha256": template["source_csv_sha256"],
                  "real_audio_checksums_sha256": digest(plans / "real_audio_checksums.json"),
                  "metadata_csv_sha256": digest(real / "additional_metadata.csv"),
                  "synthetic_manifest_sha256": digest(manifest),
                  "generator_sha256": digest(HERE / "synthetic_pcg.py"),
                  "synthetic_splitter_sha256": digest(HERE / "synthetic_splits.py"),
                  "template_sha256": digest(HERE / "real_split_template.json"),
                  "synthetic_counts_per_class": {
                      "train": preset["train_per_class"],
                      "validation": preset["validation_per_class"]},
                  "real_audit": audit}
        write_json(workspace / "data_setup.json", config)
        write_json(workspace / "setup_status.json", {"status": "passed", "mode": args.mode})
        print("PASS: prepared", workspace)
        print("No training performed; no real-test performance evaluated.")
    except BaseException as error:
        write_json(workspace / "setup_status.json", {
            "status": "failed", "mode": args.mode, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
