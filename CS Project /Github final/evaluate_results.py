"""Explicitly freeze full experiments, then evaluate selected real-test checkpoints.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: https://proceedings.mlr.press/v297/nonaka26a.html
Never invoked automatically by the training runner. Smoke/pilot test evaluation
is deliberately unsupported. Do not tune settings after inspecting test results.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment_data import load_real, setup_info
from run_experiments import environment, evaluate, validate_artifacts
from workflow import (PROTOCOLS, digest, read_json, require, source_hashes,
                      write_json)

METRICS = ("roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "ppv", "npv")
PAPER_AUROC = {"real": {"mean": 0.7165, "sd": 0.0351},
               "syn": {"mean": 0.5, "sd": 0.0},
               "syn2real": {"mean": 0.7857, "sd": 0.0074}}


def summarize(records):
    result = {}
    for protocol in PROTOCOLS:
        group = [r for r in records if r["protocol"] == protocol]
        require(len(group) == 5 and {r["run_index"] for r in group} == set(range(5)),
                "Final summary requires exactly five unique runs per protocol")
        result[protocol] = {metric: {
            "mean": float(np.mean([r["metrics"][metric] for r in group])),
            "sd": float(np.std([r["metrics"][metric] for r in group], ddof=1))}
            for metric in METRICS}
    return result


def freeze(root):
    config = read_json(root / "suite_config.json")
    require(config["settings"]["mode"] == "full", "Only full experiments can be frozen")
    require(set(config["settings"]["run_indices"]) == set(range(5)),
            "Freeze requires all five paired runs in one suite")
    require(read_json(root / "suite_status.json")["status"] == "complete",
            "Finish the complete full suite before freezing")
    require(config["provenance"]["package_source_hashes"] == source_hashes(),
            "Package code changed since training")
    marker = root / "frozen.json"
    require(not marker.exists(), "Suite is already frozen; existing marker is preserved")
    files = {"suite_config.json": digest(root / "suite_config.json"),
             "suite_status.json": digest(root / "suite_status.json")}
    for index in range(5):
        for protocol in PROTOCOLS:
            stage = root / f"run_{index:02d}" / protocol
            state = validate_artifacts(stage)
            require(state["complete"], "Incomplete stage")
            require(state["config"]["settings"] == config["settings"] and
                    state["config"]["provenance"] == config["provenance"],
                    "Stage differs from suite configuration")
            for name in ("config.json", "history.json", "best.pt", "stage_status.json"):
                path = stage / name
                files[path.relative_to(root).as_posix()] = digest(path)
    write_json(marker, {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "files": files, "test_evaluated": False})
    print("Frozen configurations and selected checkpoints. No predictions performed.")


def run_evaluation(root, workspace, device, batch_size):
    from pyhealth.models import ResNet1D

    frozen = read_json(root / "frozen.json")
    for relative, expected in frozen["files"].items():
        require(digest(root / relative) == expected, f"Frozen file changed: {relative}")
    suite = read_json(root / "suite_config.json")
    require(suite["settings"]["mode"] == "full", "Pilot/smoke test evaluation is disabled")
    require(suite["provenance"]["package_source_hashes"] == source_hashes(),
            "Code changed since training")
    current = environment(device)
    require(current["team_source_hashes"] == suite["provenance"]["environment"]["team_source_hashes"],
            "Installed team implementation differs from training")
    require(current["torch"] == suite["provenance"]["environment"]["torch"],
            "Use the same PyTorch version as training")
    setup, _, _, _ = setup_info(workspace)
    require(all(setup[k] == suite["provenance"][k] for k in (
        "real_plan_sha256", "synthetic_plan_sha256", "real_csv_sha256",
        "synthetic_manifest_sha256", "real_audio_checksums_sha256")),
            "Evaluation data plans differ")
    destination = root / "heldout"
    require(not destination.exists(), "Held-out evaluation output exists; no overwrite/repeated tuning")
    destination.mkdir()
    torch.set_num_threads(1)
    records = []
    write_json(destination / "status.json", {"status": "running", "test_evaluated": False})
    try:
        for index in range(5):
            train, _, test, _ = load_real(workspace, index)
            for protocol in PROTOCOLS:
                path = root / f"run_{index:02d}" / protocol / "best.pt"
                selected = torch.load(path, map_location="cpu", weights_only=True)
                model = ResNet1D(train, **selected["config"]["model"])
                model.load_state_dict(selected["model"])
                model.to(device)
                scores, predictions = evaluate(model, test, batch_size)
                prediction_path = destination / f"run_{index:02d}_{protocol}_predictions.json"
                write_json(prediction_path, predictions)
                record = {"run_index": index, "protocol": protocol,
                          "selected_epoch": selected["epoch"], "metrics": scores,
                          "checkpoint_sha256": digest(path)}
                records.append(record)
                write_json(destination / "partial_records.json", records)
                print(index, protocol, scores, flush=True)
                del model
            del train, test
        result = {"mode": "full", "test_evaluated": True, "records": records,
                  "summary": summarize(records), "sd_definition": "sample SD, ddof=1",
                  "threshold": 0.5, "undefined_predictive_values": "reported as zero",
                  "paper_table17_resnet18_as_auroc": PAPER_AUROC,
                  "comparison_note": "Different cohort/splits/preprocessing and no augmentation; "
                      "descriptive comparison, not an exact paper reproduction.",
                  "frozen_sha256": digest(root / "frozen.json"),
                  "evaluation_device": device}
        write_json(destination / "report.json", result)
        write_json(destination / "status.json", {"status": "complete", "test_evaluated": True})
        print("Saved final report:", destination / "report.json")
    except BaseException as error:
        write_json(destination / "status.json", {
            "status": "failed", "completed_evaluations": len(records),
            "test_evaluated": bool(records), "error": str(error),
            "note": "Inspect existing predictions; do not tune/restart silently."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Completed full training output root")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confirm-frozen", action="store_true",
                        help="Explicitly permit held-out evaluation of frozen full experiments")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.output.resolve())
    else:
        require(args.workspace is not None and args.confirm_frozen,
                "Evaluation requires --workspace and --confirm-frozen")
        require(args.batch_size > 0, "Batch size must be positive")
        run_evaluation(args.output.resolve(), args.workspace, args.device, args.batch_size)


if __name__ == "__main__":
    main()
