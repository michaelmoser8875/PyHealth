"""Small isolated workflow tests; no model training, downloads or real data.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: https://proceedings.mlr.press/v297/nonaka26a.html
"""

import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiment_data import loader, shuffled_cycles
from prepare_data import audit_real
from run_experiments import (binary_scores, check_batch,
                             save_checkpoint, validate_artifacts)
from workflow import (HERE, digest, read_json, settings, validate_real_plan,
                      write_json)


class TestExperimentPolicy(unittest.TestCase):
    def test_presets_and_explicit_overrides(self):
        full = settings("full")
        self.assertEqual(full["run_indices"], list(range(5)))
        self.assertEqual((full["epochs"], full["val_every"], full["patience"]), (500, 5, 5))
        self.assertEqual(full["batch_size"], 512)
        self.assertEqual(settings("pilot")["minutes"], 60)
        self.assertEqual(settings("smoke")["epochs"], 2)
        self.assertEqual(settings("full", batch_size=8, epochs=25)["epochs"], 25)

    def test_invalid_options_rejected(self):
        for kw in ({"batch_size": 3}, {"epochs": 0}, {"patience": -1},
                   {"minutes": float("nan")}, {"run_indices": [0, 0]},
                   {"run_indices": [5]}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                settings("pilot", **kw)

    def test_real_template_and_leakage_guard(self):
        plan = read_json(HERE / "real_split_template.json")
        validate_real_plan(plan)
        broken = copy.deepcopy(plan)
        broken["runs"][0]["train"].append(broken["runs"][0]["test"][0])
        with self.assertRaises(ValueError):
            validate_real_plan(broken)

    def test_fixed_test_across_runs(self):
        plan = read_json(HERE / "real_split_template.json")
        plan["runs"][1]["test"], plan["runs"][1]["validation"] = (
            plan["runs"][1]["validation"], plan["runs"][1]["test"])
        with self.assertRaises(ValueError):
            validate_real_plan(plan)

    def test_shuffled_cycles_and_short_final_batches(self):
        a = shuffled_cycles(7, 20, seed=3)
        self.assertEqual(a, shuffled_cycles(7, 20, seed=3))
        self.assertEqual(set(a[:7]), set(range(7)))
        self.assertEqual(set(a[7:14]), set(range(7)))
        self.assertEqual(len(a), 20)
        data = [{"signal": torch.zeros(1, 80000), "label": torch.tensor([i % 2]),
                 "patient_id": f"p{i}"} for i in range(10)]
        batches = list(loader(data, 4, indices=shuffled_cycles(10, 10, 1)))
        self.assertEqual([len(b["label"]) for b in batches], [4, 4, 2])
        for batch in batches:
            check_batch(batch)

    def test_threshold_and_zero_denominators(self):
        scores = binary_scores([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        self.assertEqual(scores["roc_auc"], 1.0)
        self.assertEqual(scores["f1"], 1.0)
        scores = binary_scores([0, 1], [0.1, 0.1])
        self.assertEqual(scores["ppv"], 0.0)
        self.assertTrue(all(np.isfinite(v) for v in scores.values()))
        self.assertEqual(binary_scores([0, 1], [0.1, 0.5])["tp"], 1)
        with self.assertRaises(ValueError):
            binary_scores([1, 1], [0.1, 0.5])


class TestSafeArtifacts(unittest.TestCase):
    def test_cohort_audit_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").mkdir()
            (root / "train/r.wav").write_bytes(b"fixture")
            (root / "additional_metadata.csv").write_text("patient_id\np\n")
            with (root / "train.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["patient_id", "AS", "N", "recording_1"])
                writer.writeheader()
                writer.writerow({"patient_id": "p", "AS": 1, "N": 0, "recording_1": "r"})
            plan = {"source_csv_sha256": digest(root / "train.csv"),
                    "patients": {"p": {"label": 1, "recordings": ["train/r.wav"]}}}
            self.assertEqual(audit_real(root, plan)["eligible_recordings"], 1)
            (root / "train/r.wav").unlink()
            with self.assertRaises(ValueError):
                audit_real(root, plan)

    def test_atomic_json_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "x.json", {"v": 1})
            self.assertEqual(read_json(root / "x.json"), {"v": 1})
            save_checkpoint(root / "x.pt", {"tensor": torch.ones(2)})
            state = torch.load(root / "x.pt", weights_only=True)
            torch.testing.assert_close(state["tensor"], torch.ones(2))
            self.assertFalse(list(root.glob("*.tmp")))

    def fixture(self, root):
        config = {"test_evaluated": False}
        history = [{"epoch": 1, "steps": 1, "real_seen": 2, "synthetic_seen": 0,
                    "validation": {"loss": 0.4}},
                   {"epoch": 2, "steps": 2, "real_seen": 4, "synthetic_seen": 0,
                    "validation": {"loss": 0.5}}]
        state = {"config": config, "history": history, "epoch": 2, "steps": 2,
                 "real_seen": 4, "synthetic_seen": 0, "complete": True, "best_loss": 0.4,
                 "model": {"weight": torch.ones(1)},
                 "optimizer": {"state": {0: {"step": torch.tensor(2.),
                    "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}}}}
        write_json(root / "config.json", config)
        write_json(root / "history.json", history)
        save_checkpoint(root / "last.pt", state)
        save_checkpoint(root / "best.pt", {"config": config, "epoch": 1,
                                           "validation": {"loss": 0.4}})
        return state

    def test_checkpoint_selection_and_counter_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)
            self.assertTrue(validate_artifacts(root)["complete"])
            state["optimizer"]["state"][0]["step"] = torch.tensor(3.)
            save_checkpoint(root / "last.pt", state)
            with self.assertRaises(ValueError):
                validate_artifacts(root)

    def test_rejects_wrong_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = self.fixture(root)
            save_checkpoint(root / "best.pt", {"config": state["config"], "epoch": 2,
                                               "validation": {"loss": 0.5}})
            with self.assertRaises(ValueError):
                validate_artifacts(root)


if __name__ == "__main__":
    unittest.main()
