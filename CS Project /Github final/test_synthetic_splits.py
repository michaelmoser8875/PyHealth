"""Tests for fixed-pool split policy; all data lives in temporary folders."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from synthetic_splits import create_split_plan, read_synthetic_run, load_synthetic_run
from test_synthetic_manifest import ManifestFixture, HAS_PYHEALTH


class TestSyntheticSplits(ManifestFixture):
    def make_plan(self, **kwargs):
        return create_split_plan(self.manifest, self.root / "runs" / "plan.json", **kwargs)

    def test_default_policy_preserves_first_and_reuses_pool(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.iterdir() if p.is_file()}
        path = self.make_plan()
        plan = json.loads(path.read_text())
        self.assertEqual(len(plan["runs"]), 5)
        self.assertEqual([r["training_seed"] for r in plan["runs"]], list(range(5)))
        self.assertEqual([r["split_seed"] for r in plan["runs"]], [None, 42, 43, 44, 45])
        for i in range(5):
            samples = read_synthetic_run(path, i, verify_checksums=True)
            self.assertEqual([len(samples[s]) for s in ("train", "validation")], [4, 4])
            ids = {s: {x["record_id"] for x in v} for s, v in samples.items()}
            self.assertFalse(ids["train"] & ids["validation"])
            self.assertEqual(ids["train"] | ids["validation"],
                             {row["recording_id"] for row in self.rows})
            if i == 0:
                for split in ids:
                    self.assertEqual(ids[split], {r["recording_id"] for r in self.rows
                                                if r["split"] == split})
        self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == h
                            for p, h in before.items()))
        self.assertFalse(list((self.root / "runs").rglob("*.wav")))

    def test_explicit_seeds_counts_and_no_preserved_run(self):
        path = self.make_plan(n_runs=2, preserve_first=False,
                             split_seeds=[101, 202], training_seeds=[7, 8],
                             train_per_class=3, validation_per_class=1)
        plan = json.loads(path.read_text())
        self.assertEqual([r["split_seed"] for r in plan["runs"]], [101, 202])
        self.assertEqual([r["training_seed"] for r in plan["runs"]], [7, 8])
        samples = read_synthetic_run(path, 1)
        self.assertEqual([len(samples[s]) for s in ("train", "validation")], [6, 2])

    def test_reproducibility_and_single_run(self):
        first = self.make_plan(n_runs=1)
        second = create_split_plan(self.manifest, first.with_name("second.json"), n_runs=1)
        self.assertEqual(json.loads(first.read_text()), json.loads(second.read_text()))
        with self.assertRaises(FileExistsError):
            create_split_plan(self.manifest, first)

    def test_bad_configuration_does_not_write_plan(self):
        options = [dict(n_runs=0), dict(n_runs=True), dict(split_seed=-1),
                   dict(split_seeds=[1]), dict(training_seeds=[0]*5),
                   dict(preserve_first="yes"), dict(train_per_class=3),
                   dict(preserve_first=False, train_per_class=3, validation_per_class=2)]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_plan(**kwargs)
            self.assertFalse((self.root / "runs").exists())

    def test_changed_source_is_rejected(self):
        path = self.make_plan()
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_synthetic_run(path)

    def test_all_runs_checked_before_subset_returned(self):
        path = self.make_plan()
        original = json.loads(path.read_text())
        for kind in ("overlap", "unknown", "duplicate", "class_count", "preserved"):
            plan = json.loads(json.dumps(original))
            run = plan["runs"][-1]
            if kind == "overlap":
                run["validation"][0] = run["train"][0]
            elif kind == "unknown":
                run["train"][0] = "missing-recording"
            elif kind == "duplicate":
                run["train"][0] = run["train"][1]
            elif kind == "class_count":
                plan["counts_per_class"]["train"] = 1
            else:
                run["preserves_original"] = True
                run["train"] = list(plan["runs"][0]["validation"])
                run["validation"] = list(plan["runs"][0]["train"])
            path.write_text(json.dumps(plan))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                read_synthetic_run(path, 0, limit_per_class=1)

    def test_limits_indices_and_relative_portability(self):
        path = self.make_plan()
        with self.assertRaises(ValueError):
            read_synthetic_run(path, 5)
        with self.assertRaises(ValueError):
            read_synthetic_run(path, True)
        with self.assertRaises(ValueError):
            read_synthetic_run(path, limit_per_class=0)
        limited = read_synthetic_run(path, 2, limit_per_class=1)
        self.assertEqual([len(v) for v in limited.values()], [2, 2])
        with tempfile.TemporaryDirectory() as other:
            moved = Path(other) / "pool"
            shutil.copytree(self.root, moved)
            samples = read_synthetic_run(moved / "runs" / "plan.json", 2,
                                         verify_checksums=True)
            self.assertTrue(all(Path(s["signal"]).is_relative_to(moved.resolve())
                                for rows in samples.values() for s in rows))


@unittest.skipUnless(HAS_PYHEALTH, "Requires PyHealth/audio in Python 3.12/3.13")
class TestRunPyHealthIntegration(ManifestFixture):
    def test_run_loader_uses_shared_train_processors(self):
        path = create_split_plan(self.manifest, self.root / "runs.json", n_runs=2)
        train, val = load_synthetic_run(path, 1, limit_per_class=1, in_memory=True)
        self.assertEqual((len(train), len(val)), (2, 2))
        self.assertIs(train.input_processors["signal"], val.input_processors["signal"])
        self.assertIs(train.output_processors["label"], val.output_processors["label"])
        self.assertEqual(tuple(train[0]["signal"].shape), (1, 80000))


if __name__ == "__main__":
    unittest.main()
