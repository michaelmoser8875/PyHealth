"""Read/verify this package's own saved checkpoints; never train or predict.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: https://proceedings.mlr.press/v297/nonaka26a.html
Only inspect local checkpoints you produced and trust (last.pt contains pickle).
"""

import argparse
import json
from pathlib import Path

import torch

from run_experiments import validate_artifacts
from workflow import PROTOCOLS, digest, read_json, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    suite = read_json(root / "suite_config.json")
    status = read_json(root / "suite_status.json")
    require(status["status"] in ("complete", "partial", "interrupted", "failed"),
            "Unrecognized suite status")
    records = []
    for index in suite["settings"]["run_indices"]:
        for protocol in PROTOCOLS:
            directory = root / f"run_{index:02d}" / protocol
            if not (directory / "last.pt").is_file():
                continue
            state = validate_artifacts(directory)
            best_path = directory / "best.pt"
            best = (torch.load(best_path, map_location="cpu", weights_only=True)
                    if best_path.exists() else None)
            require(state["config"]["settings"] == suite["settings"] and
                    state["config"]["provenance"] == suite["provenance"],
                    "Stage configuration differs from suite")
            item = {"run_index": index, "protocol": protocol,
                    "complete": state["complete"], "epoch": state["epoch"],
                    "steps": state["steps"], "real_seen": state["real_seen"],
                    "synthetic_seen": state["synthetic_seen"],
                    "elapsed_minutes": state["elapsed_seconds"] / 60,
                    "selected_epoch": best["epoch"] if best else None,
                    "validation_domain": "synthetic" if protocol == "syn" else "real",
                    "validation": best["validation"] if best else None}
            if protocol == "syn2real":
                source = root / f"run_{index:02d}/syn/best.pt"
                require(digest(source) == state["config"]["pretraining"]["sha256"],
                        "Fine-tuning source checkpoint differs")
                syn_state = validate_artifacts(source.parent)
                item["shared_pretraining_steps"] = syn_state["steps"]
                item["shared_pretraining_synthetic_seen"] = syn_state["synthetic_seen"]
                item["note"] = "Shared pretraining is reused, not a second training cost"
            records.append(item)
    if status["status"] == "complete":
        require(len(records) == 4 * len(suite["settings"]["run_indices"]) and
                all(r["complete"] for r in records), "Suite claims completion incorrectly")
    print(json.dumps({"status": status["status"], "mode": suite["settings"]["mode"],
                      "test_evaluated_by_training": False, "stages": records,
                      "warning": "Validation results only. Syn uses synthetic validation; "
                          "other protocols use real validation. Do not rank protocols "
                          "across different validation domains or claim paper reproduction."},
                     indent=2))


if __name__ == "__main__":
    main()
